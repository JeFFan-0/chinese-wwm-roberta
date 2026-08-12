"""P8 测试：情绪因子输出协议与聚合骨架。

对应 TODO §11：test_factor_time_boundary / test_factor_traceability，
以及 §10.4 的合成验收项（去重、时间边界、衰减、聚合、缺失值）。
"""
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.factor import (  # noqa: E402
    LABEL_COLUMNS,
    aggregate_daily_factor,
    check_no_label_fields_when_unknown,
    deduplicate_exact,
    factor_day,
    map_predictions,
    point_in_time_check,
    reject_low_confidence,
    time_decay_weight,
    to_raw_predictions,
    trading_window,
)

TZ8 = timezone(timedelta(hours=8))
CONFIG = {
    "config_version": "1.0",
    "research_timezone": "Asia/Shanghai",
    "time_decay_half_life_hours": 24.0,
    "source_weights": {"default": 1.0, "src_a": 2.0},
    "min_max_probability": 0.6,
    "winsorize": {"lower_quantile": 0.01, "upper_quantile": 0.99},
    "missing": {"missing_entity": "reject", "missing_time": "reject"},
}


def _dt(y, m, d, hh, mm, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=TZ8)


def _raw_rows(n=4, label_confirmed=False):
    """构造若干 raw 行：第 0、1 同实体同日盘中；第 2 盘后；第 3 缺失实体。"""
    text_ids = ["t0", "t1", "t2", "t3"]
    entities = ["E1", "E1", "E2", None]
    times = [_dt(2024, 1, 3, 10, 0), _dt(2024, 1, 3, 11, 0),
             _dt(2024, 1, 3, 16, 0), _dt(2024, 1, 3, 12, 0)]
    probs = [(0.9, 0.1), (0.8, 0.2), (0.1, 0.9), (0.7, 0.3)]
    logits = [(1.0, -1.0), (0.8, -0.8), (-1.0, 1.0), (0.5, -0.5)]
    df = to_raw_predictions(
        text_ids=text_ids,
        class_0_logit=[l[0] for l in logits],
        class_1_logit=[l[1] for l in logits],
        class_0_prob=[p[0] for p in probs],
        class_1_prob=[p[1] for p in probs],
        exit_layer=[11] * n,
        head_type="shared_frozen_head", pooling="masked_mean",
        model_hash="hash-test", config_version="1.0",
        entity_ids=entities[:n], published_at=times[:n],
        sources=["src_a", "default", "default", "default"],
        label_mapping_confirmed=label_confirmed,
    )
    return df


# --------------------------------------------------------------------------- #
# 标签未知时不输出正负字段
# --------------------------------------------------------------------------- #
def test_label_unknown_no_pos_neg_fields():
    raw = _raw_rows(label_confirmed=False)
    assert check_no_label_fields_when_unknown(raw)
    assert not set(LABEL_COLUMNS).intersection(raw.columns)
    raw_conf = _raw_rows(label_confirmed=True)
    assert set(LABEL_COLUMNS).issubset(raw_conf.columns)
    assert (raw_conf["sentiment_score"] == raw_conf["p_positive"] - raw_conf["p_negative"]).all()


# --------------------------------------------------------------------------- #
# 时间字段带时区 / 交易窗口归属
# --------------------------------------------------------------------------- #
def test_time_fields_timezone_aware():
    raw = _raw_rows()
    mapped, rejected = map_predictions(raw, label_mapping_confirmed=True, config=CONFIG)
    assert mapped["published_at_local"].dt.tz is not None  # 全部带时区
    # 无时区时间应被拒绝
    from src.factor import to_raw_predictions as trp
    bad = trp(
        text_ids=["x"], class_0_logit=[0.0], class_1_logit=[0.0],
        class_0_prob=[0.5], class_1_prob=[0.5], exit_layer=[11],
        head_type="h", pooling="p", model_hash="m", config_version="v",
        entity_ids=["E"], published_at=[datetime(2024, 1, 3, 10, 0)],  # 无时区
        sources=["s"], label_mapping_confirmed=True,
    )
    # 无时区时间在 to_research_local 时报错 → 应被捕获为缺失
    with pytest.raises(ValueError):
        map_predictions(bad, label_mapping_confirmed=True, config=CONFIG)


def test_trading_window():
    assert trading_window(_dt(2024, 1, 3, 9, 0)) == "pre"
    assert trading_window(_dt(2024, 1, 3, 12, 0)) == "intra"
    assert trading_window(_dt(2024, 1, 3, 16, 0)) == "after"


# --------------------------------------------------------------------------- #
# test_factor_time_boundary：盘后文本不进入当天收盘前因子
# --------------------------------------------------------------------------- #
def test_factor_time_boundary():
    intra = _dt(2024, 1, 3, 14, 0)     # 周三盘中 → 归 1/3
    after = _dt(2024, 1, 3, 16, 0)     # 周三盘后 → 下一交易日 1/4
    weekend = _dt(2024, 1, 6, 10, 0)   # 周六 → 下一交易日 1/8（周一）
    assert factor_day(intra)[0] == date(2024, 1, 3)
    assert factor_day(after)[0] == date(2024, 1, 4)
    assert factor_day(weekend)[0] == date(2024, 1, 8)

    raw = _raw_rows(4, label_confirmed=True)
    mapped, rejected = map_predictions(raw, label_mapping_confirmed=True, config=CONFIG)
    daily = aggregate_daily_factor(mapped, CONFIG)
    # 盘后文本 t2（E2，1/3 16:00）的 factor_day = 1/4，不会出现在 1/3 的 E1 因子里
    assert (daily["factor_day"].astype(str).str.contains("2024-01-03")).any()
    assert not set(daily["factor_day"].astype(str)).issubset({"2024-01-03"})


def test_point_in_time_check():
    raw = _raw_rows(label_confirmed=True)
    mapped, rejected = map_predictions(raw, label_mapping_confirmed=True, config=CONFIG)
    assert point_in_time_check(None, mapped)  # 所有文本 factor_day >= 发布日


# --------------------------------------------------------------------------- #
# 去重：重复文本不重复计权
# --------------------------------------------------------------------------- #
def test_dedup_not_double_counted():
    raw = _raw_rows(label_confirmed=True)
    dup = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)   # t0 重复出现
    dedup = deduplicate_exact(dup, on=["text_id"])
    assert len(dedup) == len(raw)
    assert dedup["text_id"].nunique() == len(dedup)


# --------------------------------------------------------------------------- #
# 聚合：同日同主体正确
# --------------------------------------------------------------------------- #
def test_same_day_same_entity_aggregation():
    raw = _raw_rows(label_confirmed=True)
    mapped, rejected = map_predictions(raw, label_mapping_confirmed=True, config=CONFIG)
    daily = aggregate_daily_factor(mapped, CONFIG)
    e1 = daily[(daily["entity_id"] == "E1")]
    # E1 两文本均归 1/3 → 聚合为 1 行
    e1_103 = e1[e1["factor_day"] == date(2024, 1, 3)]
    assert len(e1_103) == 1
    assert e1_103["n_texts"].iloc[0] == 2
    # 追踪到源文本 ID
    assert set(e1_103["text_ids"].iloc[0]) == {"t0", "t1"}


def test_traceability_text_ids():
    raw = _raw_rows(label_confirmed=True)
    mapped, rejected = map_predictions(raw, label_mapping_confirmed=True, config=CONFIG)
    daily = aggregate_daily_factor(mapped, CONFIG)
    all_ids = {tid for lst in daily["text_ids"] for tid in lst}
    assert all_ids.issubset(set(raw["text_id"]))  # 可回溯至源文本


# --------------------------------------------------------------------------- #
# 时间衰减方向
# --------------------------------------------------------------------------- #
def test_time_decay_direction():
    ref = _dt(2024, 1, 3, 15, 0)
    old = time_decay_weight(_dt(2024, 1, 3, 8, 0), ref, 24.0)
    recent = time_decay_weight(_dt(2024, 1, 3, 14, 0), ref, 24.0)
    assert old < recent
    assert 0 < old <= 1.0


# --------------------------------------------------------------------------- #
# 缺失值：缺失主体/时间被拒绝
# --------------------------------------------------------------------------- #
def test_missing_entity_rejected():
    raw = _raw_rows()  # t3 entity=None
    mapped, rejected = map_predictions(raw, label_mapping_confirmed=True, config=CONFIG)
    assert "t3" in set(rejected["text_id"])
    assert "t3" not in set(mapped["text_id"])


# --------------------------------------------------------------------------- #
# 低置信度拒识
# --------------------------------------------------------------------------- #
def test_low_confidence_reject():
    raw = _raw_rows()
    passed, rejected = reject_low_confidence(raw, min_max_prob=0.85)
    assert passed["max_probability"].min() >= 0.85
    assert set(rejected["text_id"]).issubset(set(raw["text_id"]))


# --------------------------------------------------------------------------- #
# 一致性：同输入同配置版本 → 相同因子结果
# --------------------------------------------------------------------------- #
def test_consistent_config_version():
    raw = _raw_rows(label_confirmed=True)
    m1, _ = map_predictions(raw, label_mapping_confirmed=True, config=CONFIG)
    m2, _ = map_predictions(raw, label_mapping_confirmed=True, config=CONFIG)
    d1 = aggregate_daily_factor(m1, CONFIG)
    d2 = aggregate_daily_factor(m2, CONFIG)
    pd.testing.assert_frame_equal(d1, d2)


# --------------------------------------------------------------------------- #
# 原始输出不可变：聚合生成新表
# --------------------------------------------------------------------------- #
def test_raw_immutable():
    raw = _raw_rows(label_confirmed=True)
    raw_before = raw.copy(deep=True)
    mapped, _ = map_predictions(raw, label_mapping_confirmed=True, config=CONFIG)
    aggregate_daily_factor(mapped, CONFIG)
    pd.testing.assert_frame_equal(raw, raw_before)
