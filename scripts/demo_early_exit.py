#!/usr/bin/env python
"""P4 演示：真正逐层执行的 Early-Exit 固定层退出与动态阈值 smoke test。

输出执行追踪日志：exit_layer / executed_layer_count / exit_reason /
layer_call_counts（证明退出后后续层确实未调用）与 latency_ms。

用法:
    conda activate 26intern
    python scripts/demo_early_exit.py [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_yaml_config  # noqa: E402
from src.models.early_exit import build_early_exit_engine  # noqa: E402
from src.models.modeling import load_tokenizer, tokenize_texts  # noqa: E402

TEXTS = [
    "北京天气怎么样，明天会下雨吗？",
    "这个项目的核心目标是提升模型的每一层利用率，而不是只用最后一层。",
    "今天股市大涨，投资者情绪明显回暖。",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    ckpt = os.path.join(ROOT, cfg["paths"]["checkpoint"])
    base_dir = os.path.join(ROOT, cfg["paths"]["base_model_dir"])

    engine = build_early_exit_engine(base_dir, ckpt, head_type="copied_layer_heads",
                                     pooling="masked_mean", device=args.device)
    tokenizer = load_tokenizer(base_dir)
    enc = tokenize_texts(TEXTS, tokenizer)
    enc = {k: v.to(args.device) for k, v in enc.items()}

    print("=" * 88)
    print(f"P4 Early-Exit 演示（head_type=copied_layer_heads, pooling=masked_mean, device={args.device}）")
    print("=" * 88)

    # ---- 1. 固定层退出 ----
    print("\n[1] 固定层退出（正式基准唯一策略）")
    baseline = engine.full_forward_baseline(**enc)
    for k in (2, 4, 8, 11):
        r = engine.run_fixed(**enc, exit_layer=k)
        eq = torch.allclose(r.logits, baseline, atol=1e-4)
        calls = r.layer_call_counts
        calls_str = " ".join(f"{c}" if c else "." for c in calls)
        print(f"  exit_layer={k:<2} executed={r.executed_layer_count:<2} "
              f"equiv_final={eq}  latency={r.latency_ms:7.1f}ms  calls=[{calls_str}]")

    # ---- 2. 动态阈值（仅 smoke test，未校准）----
    print("\n[2] 动态阈值 smoke test（batch size 1；未校准，不得用于正式部署）")
    single = {k: v[:1] for k, v in enc.items()}
    for thr, label in ((-1e9, "极低阈值（首个候选层退出）"), (1.0, "极高阈值（最后层兜底）")):
        r = engine.run_dynamic(**single, strategy="max_prob", threshold=thr,
                               candidate_layers=[2, 4, 6, 8, 10, 11])
        calls = r.layer_call_counts
        calls_str = " ".join(f"{c}" if c else "." for c in calls)
        print(f"  {label}: exit_layer={r.exit_layer} reason={r.exit_reason:<19} "
              f"executed={r.executed_layer_count:<2} calls=[{calls_str}]")

    # ---- 3. active-set（v2，恢复原顺序）----
    print("\n[3] active-set 动态退出（batch，恢复原顺序）")
    r, reasons = engine.run_active_set(**enc, strategy="max_prob", threshold=-1e9,
                                       candidate_layers=[2, 4, 6, 8, 10, 11])
    print(f"  exit_reasons={reasons}  输出顺序与输入顺序一致")
    print(f"  logits[0].tolist()={r.logits[0].tolist()}")

    print("\n结果: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
