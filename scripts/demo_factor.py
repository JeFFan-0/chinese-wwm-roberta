#!/usr/bin/env python
"""P8 演示：三层因子管线（raw → mapped → daily_factor）合成样例输出。

当前标签映射未确认，raw/mapped 不输出正负情绪字段（synthetic_only）。

用法:
    conda activate 26intern
    python scripts/demo_factor.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_yaml_config  # noqa: E402
from src.factors.factor import (  # noqa: E402
    aggregate_daily_factor,
    map_predictions,
    to_raw_predictions,
)

TZ8 = timezone(timedelta(hours=8))


def main() -> int:
    cfg = load_yaml_config(os.path.join(ROOT, "configs", "factor.yaml"))["aggregation"]
    label_confirmed = cfg.get("label_mapping_confirmed", False)

    # 合成预测（伪概率，仅演示管线）
    n = 6
    raw = to_raw_predictions(
        text_ids=[f"t{i}" for i in range(n)],
        class_0_logit=[0.5, -1.0, 0.2, -0.4, 0.8, -0.9],
        class_1_logit=[-0.5, 1.0, -0.2, 0.4, -0.8, 0.9],
        class_0_prob=[0.62, 0.12, 0.55, 0.40, 0.69, 0.14],
        class_1_prob=[0.38, 0.88, 0.45, 0.60, 0.31, 0.86],
        exit_layer=[11] * n,
        head_type="shared_frozen_head", pooling="masked_mean",
        model_hash="synthetic-hash", config_version=cfg["config_version"],
        entity_ids=["E1", "E1", "E2", "E2", "E3", None],
        published_at=[
            datetime(2024, 1, 3, 10, 0, tzinfo=TZ8),
            datetime(2024, 1, 3, 16, 0, tzinfo=TZ8),   # 盘后 → 下一交易日
            datetime(2024, 1, 3, 11, 0, tzinfo=TZ8),
            datetime(2024, 1, 4, 14, 0, tzinfo=TZ8),
            datetime(2024, 1, 3, 9, 30, tzinfo=TZ8),
            datetime(2024, 1, 3, 12, 0, tzinfo=TZ8),   # 缺失 entity
        ],
        sources=["src_a", "default", "default", "default", "default", "default"],
        label_mapping_confirmed=label_confirmed,
    )
    print(f"label_mapping_confirmed={label_confirmed}  （当前 False → 无正负情绪字段）")
    print("\n=== 第 1 层：raw_predictions（不可变）===")
    print(raw.to_string(index=False))

    mapped, rejected = map_predictions(raw, label_mapping_confirmed=label_confirmed, config=cfg)
    print("\n=== 第 2 层：mapped_predictions（含交易时间归属）===")
    print(mapped[["text_id", "entity_id", "published_at_local", "factor_day",
                  "trading_window", "source_weight", "time_decay_weight"]].to_string(index=False))
    if label_confirmed:
        print(mapped[["text_id", "sentiment_score"]].to_string(index=False))
    print("\n被拒绝（缺失主体/时间）:",
          list(rejected["text_id"]) if not rejected.empty else "无")

    print("\n=== 第 3 层：daily_factor（主体-交易日聚合）===")
    if label_confirmed:
        daily = aggregate_daily_factor(mapped, cfg)
    else:
        # 未确认标签：用 logit_margin 作为占位信号做聚合骨架演示
        daily = aggregate_daily_factor(mapped, cfg, sentiment_col="logit_margin_1_minus_0")
    print(daily[["entity_id", "factor_day", "n_texts", "weighted_sentiment",
                 "winsorized_sentiment", "standardized_sentiment"]].to_string(index=False))

    out = os.path.join(ROOT, "reports", "tables", "factor_demo.csv")
    daily.to_csv(out, index=False)
    print(f"\n已写入 {out}（合成样例，synthetic_only）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
