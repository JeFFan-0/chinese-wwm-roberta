#!/usr/bin/env python
"""P2 heads 初始化清单与参数量报告。

输出 reports/tables/heads_init_report.csv：
    head_type, init_strategy, seed, param_count, storage_bytes, backbone_frozen, pooling

用法:
    conda activate 26intern
    python scripts/report_heads.py
"""
from __future__ import annotations

import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_yaml_config  # noqa: E402
from src.probes.heads import build_layer_heads_model  # noqa: E402


def main() -> int:
    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    heads_cfg = load_yaml_config(os.path.join(ROOT, "configs", "heads.yaml"))
    ckpt = os.path.join(ROOT, cfg["paths"]["checkpoint"])
    base_dir = os.path.join(ROOT, cfg["paths"]["base_model_dir"])

    enabled = heads_cfg["heads"]["enabled"]
    random_seed = heads_cfg["heads_spec"]["random_layer_heads"]["seed"]
    pooling = "masked_mean"  # heads 诊断默认 pooling（原模型 pooling 未确认）

    model = build_layer_heads_model(
        base_dir, ckpt, heads_enabled=enabled, pooling=pooling,
        pooling_confirmed=cfg["pooling_confirmed"], random_seed=random_seed,
        model_hash="report", device="cpu",
    )
    summary = model.head_parameter_summary()
    frozen = model.verify_backbone_frozen()

    init_strategy = {
        "original_final_head": "copy_from_fc(read-only)",
        "shared_frozen_head": "copy_from_fc(frozen, shared)",
        "copied_layer_heads": "copy_from_fc",
        "random_layer_heads": f"random(seed={random_seed})",
        "normalized_layer_heads": "copy_from_fc + LayerNorm/Dropout",
    }

    rows = []
    for name in enabled:
        rows.append({
            "head_type": name,
            "init_strategy": init_strategy[name],
            "seed": random_seed if name == "random_layer_heads" else "",
            "param_count": summary[name]["param_count"],
            "storage_bytes": summary[name]["storage_bytes"],
            "backbone_frozen": frozen,
            "pooling": pooling,
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "reports", "tables"), exist_ok=True)
    out = os.path.join(ROOT, "reports", "tables", "heads_init_report.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nbackbone_frozen={frozen}  pooling={pooling}")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
