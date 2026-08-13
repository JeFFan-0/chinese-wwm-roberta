"""情绪因子输出协议与聚合骨架（P8）。

三层数据模型（§10.3），避免覆盖原始结果：
1. ``raw_predictions``：逐文本原始 logits/probabilities（不可变）；
2. ``mapped_predictions``：增加标签含义、主体、交易时间归属（标签映射确认后才有
   p_negative/p_positive/sentiment_score）；
3. ``daily_factor``：主体-交易日级聚合因子。

所有中间表保存 model_hash 与聚合配置版本。标签映射未知时**不输出**正负情绪字段。
时间字段带时区并通过 point-in-time 检查：盘后/周末文本归属到下一交易日，
不会进入当天收盘前因子。

本模块只搭建函数与配置，不确定最终参数；合成测试只验证机制正确性。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    _RESEARCH_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # noqa: BLE001  # 无 tzdata 时退回固定 UTC+8
    _RESEARCH_TZ = __import__("datetime").timezone(timedelta(hours=8))

RAW_COLUMNS = [
    "text_id", "entity_id", "published_at", "source",
    "class_0_logit", "class_1_logit", "class_0_prob", "class_1_prob",
    "logit_margin_1_minus_0", "max_probability", "entropy",
    "exit_layer", "head_type", "pooling", "model_hash", "config_version",
]
LABEL_COLUMNS = ["p_negative", "p_positive", "sentiment_score"]


def research_timezone():
    return _RESEARCH_TZ


# --------------------------------------------------------------------------- #
# 文本级原始输出（§10.1）
# --------------------------------------------------------------------------- #
def to_raw_predictions(
    text_ids: Sequence[str],
    class_0_logit: Sequence[float],
    class_1_logit: Sequence[float],
    class_0_prob: Sequence[float],
    class_1_prob: Sequence[float],
    exit_layer: Sequence[int],
    head_type: str,
    pooling: str,
    model_hash: str,
    config_version: str,
    entity_ids: Optional[Sequence[Optional[str]]] = None,
    published_at: Optional[Sequence[Optional[datetime]]] = None,
    sources: Optional[Sequence[Optional[str]]] = None,
    label_mapping_confirmed: bool = False,
) -> pd.DataFrame:
    """构造 raw_predictions 表。

    label_mapping_confirmed=False 时（当前阶段）**不输出** p_negative/p_positive，
    满足"标签未知时不输出正负情绪字段"。
    """
    n = len(text_ids)
    if entity_ids is None:
        entity_ids = [None] * n
    if published_at is None:
        published_at = [None] * n
    if sources is None:
        sources = [None] * n
    rows = []
    for i in range(n):
        row = {
            "text_id": text_ids[i],
            "entity_id": entity_ids[i],
            "published_at": published_at[i],
            "source": sources[i],
            "class_0_logit": class_0_logit[i],
            "class_1_logit": class_1_logit[i],
            "class_0_prob": class_0_prob[i],
            "class_1_prob": class_1_prob[i],
            "logit_margin_1_minus_0": class_1_logit[i] - class_0_logit[i],
            "max_probability": max(class_0_prob[i], class_1_prob[i]),
            "entropy": _entropy(class_0_prob[i], class_1_prob[i]),
            "exit_layer": int(exit_layer[i]),
            "head_type": head_type,
            "pooling": pooling,
            "model_hash": model_hash,
            "config_version": config_version,
        }
        if label_mapping_confirmed:
            p_neg, p_pos = class_0_prob[i], class_1_prob[i]
            row.update({
                "p_negative": p_neg,
                "p_positive": p_pos,
                "sentiment_score": p_pos - p_neg,
            })
        rows.append(row)
    cols = RAW_COLUMNS + (LABEL_COLUMNS if label_mapping_confirmed else [])
    return pd.DataFrame(rows, columns=cols)


def _entropy(p0: float, p1: float) -> float:
    ps = [p for p in (p0, p1) if p > 0]
    return -sum(p * np.log(p) for p in ps)


# --------------------------------------------------------------------------- #
# 时间与交易日归属（point-in-time）
# --------------------------------------------------------------------------- #
def to_research_local(dt: datetime) -> datetime:
    """统一到研究时区。"""
    if dt.tzinfo is None:
        raise ValueError(f"published_at 必须带时区: {dt}")
    return dt.astimezone(_RESEARCH_TZ)


def is_trading_day(d: date) -> bool:
    """简化的交易日判定：仅跳过周六/周日（无节假日日历，待数据到位扩展）。"""
    return d.weekday() < 5


def trading_window(local_dt: datetime, pre_open: time = time(9, 30),
                   close: time = time(15, 0)) -> str:
    """盘前 / 盘中 / 盘后归属。"""
    t = local_dt.time()
    if t < pre_open:
        return "pre"
    if t <= close:
        return "intra"
    return "after"


def _next_trading_day(d: date) -> date:
    nd = d + timedelta(days=1)
    while not is_trading_day(nd):
        nd += timedelta(days=1)
    return nd


def factor_day(local_dt: datetime, close: time = time(15, 0)) -> Tuple[date, str]:
    """返回 (factor_day, window)。

    point-in-time 规则：
    - 周末/盘后文本归属到下一交易日（不会进入当天收盘前因子）；
    - 盘中/盘前文本归属到当日。
    """
    d = local_dt.date()
    w = trading_window(local_dt, close=close)
    if not is_trading_day(d) or w == "after":
        return _next_trading_day(d), w
    return d, w


# --------------------------------------------------------------------------- #
# 去重 / 权重 / 拒识
# --------------------------------------------------------------------------- #
def deduplicate_exact(rows: pd.DataFrame, on: Sequence[str] = ("text_id",)) -> pd.DataFrame:
    """精确去重：同 (text_id) 仅保留第一条；返回去重后的表。"""
    return rows.drop_duplicates(subset=list(on), keep="first").reset_index(drop=True)


def source_weight(row_source: Optional[str], source_weights: Dict[str, float]) -> float:
    return source_weights.get(row_source or "", source_weights.get("default", 1.0))


def time_decay_weight(published: datetime, reference: datetime,
                      half_life_hours: float = 24.0) -> float:
    """时间衰减：越旧权重越小，0.5^(age / half_life)。"""
    age_hours = (reference - published).total_seconds() / 3600.0
    return float(0.5 ** (age_hours / half_life_hours))


def reject_low_confidence(rows: pd.DataFrame, min_max_prob: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """低置信度拒识：返回 (通过, 被拒)。阈值未校准，仅占位。"""
    if "max_probability" not in rows.columns:
        raise ValueError("raw_predictions 缺少 max_probability")
    passed = rows[rows["max_probability"] >= min_max_prob].reset_index(drop=True)
    rejected = rows[rows["max_probability"] < min_max_prob].reset_index(drop=True)
    return passed, rejected


def map_predictions(
    raw: pd.DataFrame,
    label_mapping_confirmed: bool,
    config: Dict,
    reference_time: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从 raw_predictions 构造 mapped_predictions + rejected。

    流程：按配置去重 → 缺失主体/时间按各自规则拒绝 → 低置信度拒识 → 交易时间归属。
    标签未确认时不加正负字段；输出会通过 point-in-time 检查。
    """
    missing = config.get("missing", {})
    missing_entity_rule = missing.get("missing_entity", "reject")
    missing_time_rule = missing.get("missing_time", "reject")
    if missing_entity_rule != "reject" or missing_time_rule != "reject":
        raise NotImplementedError("当前仅实现 reject 规则（missing_entity/missing_time）")

    work = raw.copy()
    # 1) 去重：按 config dedup.exact_text_entity 规则
    dedup_cfg = config.get("dedup", {})
    if dedup_cfg.get("exact_text_entity") == "drop":
        work = deduplicate_exact(work, on=["text_id"])

    # 2) 缺失主体/时间：各自按规则拒绝
    reject_entity = work["entity_id"].isna()
    reject_time = work["published_at"].isna()
    reject_mask = reject_entity | reject_time
    rejected = work[reject_mask].copy()

    def _missing_reason(row):
        e, t = row["entity_id"], row["published_at"]
        e_missing = e is None or (isinstance(e, float) and pd.isna(e))
        t_missing = t is None or pd.isna(t)
        if e_missing and t_missing:
            return "missing_entity_and_time"
        if e_missing:
            return "missing_entity"
        return "missing_time"

    rejected["reject_reason"] = rejected.apply(_missing_reason, axis=1)
    mapped = work[~reject_mask].copy()

    # 3) 低置信度拒识（阈值占位，未校准）
    min_max_prob = config.get("min_max_probability")
    if min_max_prob is not None:
        mapped, low_conf = reject_low_confidence(mapped, min_max_prob)
        low_conf["reject_reason"] = "low_confidence"
        rejected = pd.concat([rejected, low_conf], ignore_index=True)

    if mapped.empty:
        return mapped, rejected
    mapped["published_at_local"] = mapped["published_at"].apply(to_research_local)
    if reference_time is None:
        reference_time = mapped["published_at_local"].max()

    fd = mapped["published_at_local"].apply(lambda dt: factor_day(dt))
    mapped["factor_day"] = [x[0] for x in fd]
    mapped["trading_window"] = [x[1] for x in fd]
    mapped["source_weight"] = mapped["source"].apply(
        lambda s: source_weight(s, config.get("source_weights", {"default": 1.0})))
    mapped["time_decay_weight"] = mapped["published_at_local"].apply(
        lambda dt: time_decay_weight(dt, reference_time,
                                     config.get("time_decay_half_life_hours", 24.0)))

    if label_mapping_confirmed:
        # 标签映射确认后，从 class 概率推导正负字段（raw 恒含 class_0/1_prob）
        mapped["p_negative"] = mapped["class_0_prob"]
        mapped["p_positive"] = mapped["class_1_prob"]
        mapped["sentiment_score"] = mapped["p_positive"] - mapped["p_negative"]

    # 4) point-in-time 检查：任何文本不得进入早于其发布日的因子
    if not point_in_time_check(None, mapped):
        raise ValueError("point-in-time 检查失败：存在文本进入早于发布日的因子")
    return mapped, rejected


# --------------------------------------------------------------------------- #
# 主体-交易日聚合（§10.3 第三层）
# --------------------------------------------------------------------------- #
def aggregate_daily_factor(
    mapped: pd.DataFrame,
    config: Dict,
    sentiment_col: str = "sentiment_score",
) -> pd.DataFrame:
    """在 mapped_predictions 上聚合 daily_factor（按 entity_id × factor_day）。

    权重 = source_weight × time_decay_weight；同一主体同一天加权平均。
    """
    if sentiment_col not in mapped.columns:
        raise ValueError(f"mapped_predictions 缺少 {sentiment_col}（标签映射未确认？）")
    if mapped.empty:
        return pd.DataFrame(columns=[
            "entity_id", "factor_day", "n_texts", "weighted_sentiment",
            "winsorized_sentiment", "standardized_sentiment",
            "model_hash", "config_version"])

    # 校验 model_hash / config_version 一致性（不得混用不同版本）
    if "model_hash" in mapped.columns and mapped["model_hash"].nunique() > 1:
        raise ValueError(f"mapped_predictions 混用多个 model_hash: {sorted(mapped['model_hash'].unique())}")
    if "config_version" in mapped.columns and mapped["config_version"].nunique() > 1:
        raise ValueError("mapped_predictions 混用多个 config_version")

    g = mapped.groupby(["entity_id", "factor_day"])

    def _agg(df: pd.DataFrame) -> pd.Series:
        wts = df["source_weight"] * df["time_decay_weight"]
        wsum = float(wts.sum())
        if wsum <= 0:                       # 权重和为零 → 退化为等权均值
            weighted = float(df[sentiment_col].mean())
        else:
            weighted = float(np.average(df[sentiment_col], weights=wts))
        return pd.Series({
            "n_texts": len(df),
            "weighted_sentiment": weighted,
            "text_ids": list(df["text_id"]),
        })

    out = g.apply(_agg, include_groups=False).reset_index()

    out["model_hash"] = mapped["model_hash"].iloc[0] if "model_hash" in mapped else ""
    out["config_version"] = mapped["config_version"].iloc[0] if "config_version" in mapped else ""
    out["winsorized_sentiment"] = winsorize_series(out["weighted_sentiment"], config)
    out["standardized_sentiment"] = cross_sectional_standardize(out["winsorized_sentiment"])
    return out


def winsorize_series(s: pd.Series, config: Dict) -> pd.Series:
    lo = config.get("winsorize", {}).get("lower_quantile", 0.01)
    hi = config.get("winsorize", {}).get("upper_quantile", 0.99)
    return s.clip(s.quantile(lo), s.quantile(hi))


def cross_sectional_standardize(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or std != std:
        return s * 0.0
    return (s - s.mean()) / std


# --------------------------------------------------------------------------- #
# point-in-time 检查
# --------------------------------------------------------------------------- #
def point_in_time_check(daily: pd.DataFrame, mapped: pd.DataFrame) -> bool:
    """校验：factor_day 的因子只用不晚于该日收盘的文本。

    返回 True 表示通过。规则：mapped 中 factor_day <= 文本所属日收盘时已知。
    """
    for _, r in mapped.iterrows():
        # 文本发布日的收盘后因子归属在 factor_day（可能是下一交易日）。
        # 文本绝不会进入早于其发布日期的 factor_day 因子。
        pub_day = to_research_local(r["published_at"]).date()
        if r["factor_day"] < pub_day:
            return False
    return True


def check_no_label_fields_when_unknown(raw: pd.DataFrame) -> bool:
    """标签未知时不得含正负字段。"""
    return not set(LABEL_COLUMNS).intersection(raw.columns)
