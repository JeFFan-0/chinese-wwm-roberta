#!/usr/bin/env python
"""P1 smoke test：三种 pooling 的完整二分类推理候选。

固定小集合覆盖：短文本、长文本、空白/近空、中英文数字标点、
两条 check.ipynb 现有示例、组批 padding 的等长/变长文本。

只验证工程稳定性（shape/确定性/无 NaN/padding 不变性），
不评价情绪语义是否正确，也不把 class_0/1 命名为 positive/negative。

用法:
    conda activate 26intern
    python scripts/smoke_inference.py [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.modeling import build_candidate, load_tokenizer, tokenize_texts  # noqa: E402

# 固定 smoke-test 文本集合（不评价情绪语义）
SMOKE_TEXTS = [
    # 短文本
    "好",
    "糟糕透了",
    # 长文本（> 128 字符，验证截断稳定）
    "这是一个很长的测试句子。" * 20,
    # 空白 / 近空文本：明确处理，不应产生 NaN
    "",
    "   ",
    "。",
    # 中英文、数字、标点混合
    "V2.0 版本发布了，2024/01/01 上线的功能你觉得怎么样?",
    # check.ipynb 现有两条示例
    "北京天气怎么样，明天会下雨吗？",
    "这个项目的核心目标是提升模型的每一层利用率，而不是只用最后一层。",
    # 中文长句
    "模型的能力不仅体现在最后一层的输出上，每一层都在逐步构建对输入的理解。",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 三种 pooling 的 smoke test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    from src.config import load_yaml_config

    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    ckpt = os.path.join(ROOT, cfg["paths"]["checkpoint"])
    base_dir = os.path.join(ROOT, cfg["paths"]["base_model_dir"])

    print("=" * 80)
    print("P1 smoke test：三种 pooling 的完整二分类推理候选")
    print(f"device={args.device}  max_length={args.max_length}  pooling_confirmed={cfg['pooling_confirmed']}")
    print("=" * 80)

    tokenizer = load_tokenizer(base_dir)
    poolings = ["cls", "pooler", "masked_mean"]

    # 1. 单条（逐条）推理，验证每文本结果与 batch 一致
    single_rows = []
    batch_rows = []
    for pooling in poolings:
        t0 = time.time()
        model = build_candidate(base_dir, ckpt, pooling=pooling,
                                pooling_confirmed=cfg["pooling_confirmed"],
                                device=args.device)
        model.eval()
        build_ms = (time.time() - t0) * 1000
        print(f"\n[pooling={pooling}] 构建模型用时 {build_ms:.0f}ms")

        with torch.no_grad():
            # 逐条推理
            for i, text in enumerate(SMOKE_TEXTS):
                enc = tokenize_texts([text], tokenizer, max_length=args.max_length)
                enc = {k: v.to(args.device) for k, v in enc.items()}
                out = model(**enc)
                row = {
                    "text_id": i,
                    "pooling": pooling,
                    "text": text[:40],
                    "class_0_logit": out.class_0_logit.item(),
                    "class_1_logit": out.class_1_logit.item(),
                    "class_0_prob": out.class_0_prob.item(),
                    "class_1_prob": out.class_1_prob.item(),
                    "logit_margin_1_minus_0": out.logit_margin_1_minus_0.item(),
                }
                single_rows.append(row)

            # 组批推理（等长 padding），验证与单条一致
            enc = tokenize_texts(SMOKE_TEXTS, tokenizer, max_length=args.max_length)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            out = model(**enc)
            for i, text in enumerate(SMOKE_TEXTS):
                batch_rows.append({
                    "text_id": i,
                    "pooling": pooling,
                    "class_0_logit": out.class_0_logit[i].item(),
                    "class_1_logit": out.class_1_logit[i].item(),
                    "class_0_prob": out.class_0_prob[i].item(),
                    "class_1_prob": out.class_1_prob[i].item(),
                    "logit_margin_1_minus_0": out.logit_margin_1_minus_0[i].item(),
                })

    # 2. 对比单条 vs batch
    print("\n[单条 vs batch 一致性]")
    sdf = pd.DataFrame(single_rows)
    bdf = pd.DataFrame(batch_rows)
    merged = sdf.merge(bdf, on=["text_id", "pooling"], suffixes=("_single", "_batch"))
    merged["logit_diff"] = (merged["class_0_logit_single"] - merged["class_0_logit_batch"]).abs()
    max_diff = merged["logit_diff"].max()
    print(f"  class_0_logit 最大绝对差: {max_diff:.2e}  {'OK' if max_diff < 1e-4 else 'FAIL'}")

    # 3. 概率有效性
    print("\n[概率有效性]")
    prob_cols = ["class_0_prob", "class_1_prob"]
    all_probs = pd.concat([sdf[prob_cols], bdf[prob_cols]])
    finite = all_probs.notna().all().all() and all_probs.isin([float("inf"), float("-inf")]).sum().sum() == 0
    row_sum = (sdf["class_0_prob"] + sdf["class_1_prob"]).abs()
    sum_ok = (row_sum - 1.0).abs().max() < 1e-5
    print(f"  所有概率有限: {finite}  行和≈1: {sum_ok}")

    # 4. 逐行输出三 pooling 对照（打印示例）
    print("\n[smoke 输出示例（前 6 条 × 3 pooling，class_0/1）]")
    print(sdf[["text_id", "text", "pooling", "class_0_prob", "class_1_prob"]]
          .head(6 * 3).to_string(index=False))

    # 5. 落盘
    os.makedirs(os.path.join(ROOT, "reports", "tables"), exist_ok=True)
    out_csv = os.path.join(ROOT, "reports", "tables", "smoke_inference.csv")
    merged[["text_id", "pooling", "class_0_logit_single", "class_1_logit_single",
            "class_0_prob_single", "class_1_prob_single", "logit_margin_1_minus_0_single",
            "logit_diff"]].to_csv(out_csv, index=False)
    print(f"\n已写入 {out_csv}")

    ok = max_diff < 1e-4 and finite and sum_ok
    print(f"\n结果: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
