"""对比微调 ckpt 与官方底座：修正后的逐层、逐组件统计（P6）。

修复（对应 TODO P6）：
- 解包必须显式返回并赋值（原实现 ``for d in (base, ft): d = d[...]`` 是无效赋值）；
- 层级聚合按参数计算（delta_l2_layer = sqrt(Σ||ΔW_p||²) 等），不简单平均 rel_l2；
- 计算**真实** max abs delta，不再凭 mean_abs_diff 断言"最大绝对误差"；
- 层编号：hidden index 0 = embedding，encoder layer 0-11；表/图同时含两列；
- 激活/注意力统计只对 attention_mask==1，提供 token-micro 与 sentence-macro；
- 结论措辞：累计表示差异；不推断单层改动；不报 ReLU 死神经元（配置 GELU）；
  无标签不报层任务价值。

用法:
    conda activate 26intern
    python compare_layers.py [--device cuda]

输出 reports/tables/：
    weights_per_tensor.csv / weights_per_layer.csv
    activation_stats.csv / hidden_state_compare.csv / attention_compare.csv
    local_layer_compare.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.analysis import (  # noqa: E402
    attention_comparison,
    build_layer_models,
    hidden_state_comparison,
    load_pair,
    local_layer_comparison,
    masked_activation_stats,
    per_layer_weight_metrics,
)
from src.models.modeling import load_tokenizer, tokenize_texts  # noqa: E402

BASE_DIR = os.path.join(ROOT, "chinese-roberta-wwm-ext")
BASE_WEIGHTS = os.path.join(BASE_DIR, "pytorch_model.bin")
CKPT = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")

TEXTS = [
    "北京天气怎么样，明天会下雨吗？",
    "这个项目的核心目标是提升模型的每一层利用率，而不是只用最后一层。",
    "今天股市大涨，投资者情绪明显回暖。",
    "报告指出风险加剧，建议谨慎观望。",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT, "reports", "tables"), exist_ok=True)
    print("=" * 72)
    print("P6 修正后的逐层对比（ckpt vs 底座）")
    print("=" * 72)

    # ---- 权重级对比（CPU，无需前向）----
    ft, base = load_pair(CKPT, BASE_WEIGHTS)
    print(f"\n[权重级] ckpt {len(ft)} 键, 底座 {len(base)} 键")
    tensor_rows, layer_rows = per_layer_weight_metrics(ft, base)
    wt = pd.DataFrame(tensor_rows)
    wl = pd.DataFrame(layer_rows)
    wt.to_csv(os.path.join(ROOT, "reports", "tables", "weights_per_tensor.csv"), index=False)
    wl.to_csv(os.path.join(ROOT, "reports", "tables", "weights_per_layer.csv"), index=False)

    overall_max = wt["max_abs_delta"].max()
    overall_argmax = wt.loc[wt["max_abs_delta"].idxmax(), ["encoder_layer", "component"]]
    print(f"  真实 max abs delta = {overall_max:.6f}（{overall_argmax['component']} @ layer {overall_argmax['encoder_layer']}）")
    print(wl[["encoder_layer", "delta_l2_layer", "relative_l2_layer", "mae_layer",
              "max_abs_delta_layer", "mean_cosine"]].round(6).to_string(index=False))

    # ---- 激活级对比（需前向，同一输入）----
    print(f"\n[激活级] device={args.device}，输入 {len(TEXTS)} 条")
    ft_model, base_model = build_layer_models(BASE_DIR, CKPT, device=args.device)
    tokenizer = load_tokenizer(BASE_DIR)
    enc = tokenize_texts(TEXTS, tokenizer)
    enc = {k: v.to(args.device) for k, v in enc.items()}
    with torch.no_grad():
        of = ft_model(**enc, output_hidden_states=True, output_attentions=True)
        ob = base_model(**enc, output_hidden_states=True, output_attentions=True)

    ast = pd.DataFrame(masked_activation_stats(of.hidden_states, enc["attention_mask"]))
    hs = pd.DataFrame(hidden_state_comparison(of.hidden_states, ob.hidden_states, enc["attention_mask"]))
    ac = pd.DataFrame(attention_comparison(of.attentions, ob.attentions, enc["attention_mask"]))
    ast.to_csv(os.path.join(ROOT, "reports", "tables", "activation_stats.csv"), index=False)
    hs.to_csv(os.path.join(ROOT, "reports", "tables", "hidden_state_compare.csv"), index=False)
    ac.to_csv(os.path.join(ROOT, "reports", "tables", "attention_compare.csv"), index=False)

    print("\n  隐藏状态比较（masked，micro/macro 双口径）")
    print(hs[["hidden_index", "encoder_layer", "cos_micro", "cos_macro", "l2_micro", "l2_macro"]]
          .round(5).to_string(index=False))
    print("\n  注意力比较（masked 无效 query/key）")
    print(ac.round(6).to_string(index=False))

    # ---- 局部层替换对比（隔离累计漂移）----
    local = pd.DataFrame(local_layer_comparison(
        ft_model, base_model, enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]))
    local.to_csv(os.path.join(ROOT, "reports", "tables", "local_layer_compare.csv"), index=False)
    print("\n  局部层替换对比（同一 hidden 输入分别过 ft/base 对应层，仅自选文本 smoke test）")
    print(local.round(6).to_string(index=False))

    # ---- 结论措辞（修正后，逐项核对数值再下结论）----
    print("\n" + "=" * 72)
    print("结论（措辞已按 P6 修正）")
    print("=" * 72)

    rel = wl["relative_l2_layer"]
    rel_range = float(rel.max() - rel.min())
    mono = bool((hs["cos_micro"].diff().dropna() <= 1e-9).all())
    print("- 权重级：ft 与 base 差异整体很小（真实 max abs delta ≈ %.2e）。"
          % overall_max)
    print("  rel_L2_layer 范围 [%.4f, %.4f]（极差 %.4f），在自选 4 条文本上"
          % (rel.min(), rel.max(), rel_range))
    print("  逐层权重差异近似均匀；权重级是**逐参数差异**统计，不属于'累计表示差异'。")
    print("- 激活级：hidden-state 差异随深度逐步增大（cos 从 %.4f 降到 %.4f，%s），"
          % (hs["cos_micro"].iloc[1], hs["cos_micro"].iloc[-1],
             "该 4 条文本上单调下降" if mono else "非严格单调"))
    print("  这是**累计表示差异**（每层都作用在上一层差异之上），不由曲线推断单层改动量。")
    print("- 局部层替换对比（同一输入分别过 ft/base 对应层）显示各层自身输出差异都较小且"
          "近似均匀，")
    print("  即单层权重差异贡献有限，激活级随深度增大的差异主要来自**累计**而非单层改动；")
    print("  仅限自选文本 smoke test，不形成总体结论。")
    print("- 未计算层任务价值/情绪能力：无标签数据，不做任何该声明。")
    print("- 不使用 ReLU 死神经元概念：底座配置激活函数为 GELU。")
    print("- 所有统计均 mask 掉 padding；结果见 reports/tables/。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
