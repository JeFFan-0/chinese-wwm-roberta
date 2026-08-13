"""数据协议与加载器（P7 §9.1-9.2）。

最小训练字段：id,text,label；情绪因子扩展：entity_id,published_at,source。

- ``load_dataset`` 支持 CSV 与 JSONL；
- ``validate_dataset`` 检测重复 ID、空文本、非法 label、无时区时间；
- ``split_dataset`` 支持指定 seed，train/dev/test 严格不重叠；
- ``make_dataloader`` 动态 padding，保留样本 ID。

本模块只验证管线工程正确性；合成数据结果不提交为模型表现结论。
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import load_yaml_config

DATETIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?(\.\d+)?"
    r"(Z|[+-]\d{2}:?\d{2})$"
)


@dataclass
class Sample:
    id: str
    text: str
    label: Optional[int] = None
    entity_id: Optional[str] = None
    published_at: Optional[datetime] = None
    source: Optional[str] = None


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# --------------------------------------------------------------------------- #
# 时间解析（要求时区）
# --------------------------------------------------------------------------- #
def parse_datetime_utc(value: Optional[str]) -> Tuple[Optional[datetime], Optional[str]]:
    """解析 ISO 时间（要求带时区），**统一转换为 UTC**，并保留小数秒。

    返回 (datetime, error)。
    """
    if value is None or str(value).strip() == "":
        return None, None
    s = str(value).strip()
    m = DATETIME_RE.match(s)
    if not m:
        return None, "invalid_datetime_format"
    year, month, day, hh, mm = (int(m.group(i)) for i in (1, 2, 3, 4, 5))
    ss = int(m.group(6) or 0)
    frac = m.group(7) or ""            # 可选小数秒 ".123456"
    micro = int(frac.lstrip(".").ljust(6, "0") or "0") if frac else 0
    tz_str = m.group(8)                # group 7 是小数秒，时区是 group 8
    if tz_str == "Z":
        tz = timezone.utc
    else:
        sign = 1 if tz_str[0] == "+" else -1
        tz_clean = tz_str[1:].replace(":", "")
        tz = timezone(sign * timedelta(hours=int(tz_clean[:2]), minutes=int(tz_clean[2:4])))
    try:
        dt = datetime(year, month, day, hh, mm, ss, micro, tzinfo=tz)
        return dt.astimezone(timezone.utc), None   # 统一到 UTC
    except ValueError:
        return None, "invalid_datetime_value"


# --------------------------------------------------------------------------- #
# 加载
# --------------------------------------------------------------------------- #
def _row_to_sample(row: Dict[str, str], label_map: Dict[str, str]) -> Sample:
    dt, _ = parse_datetime_utc(row.get("published_at"))
    label_raw = row.get("label")
    label = int(label_raw) if label_raw not in (None, "") else None
    return Sample(
        id=str(row.get("id", "")).strip(),
        text=str(row.get("text", "")),
        label=label,
        entity_id=row.get("entity_id"),
        published_at=dt,
        source=row.get("source"),
    )


def load_dataset(path: str, label_map: Optional[Dict[str, str]] = None) -> List[Sample]:
    """加载 CSV 或 JSONL 数据集。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    label_map = label_map or {"0": "unknown", "1": "unknown"}
    if path.endswith(".jsonl"):
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    else:
        with open(path, encoding="utf-8", newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    return [_row_to_sample(r, label_map) for r in rows]


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def validate_dataset(
    samples: Sequence[Sample],
    label_map: Optional[Dict[str, str]] = None,
    require_label: bool = True,
) -> ValidationResult:
    """检测重复 ID、空文本、非法 label、无时区时间等。"""
    label_map = label_map or {"0": "unknown", "1": "unknown"}
    result = ValidationResult()
    seen: Dict[str, int] = {}
    valid_labels = {int(k) for k in label_map}

    for s in samples:
        if not s.id:
            result.errors.append("empty_id")
            continue
        if s.id in seen:
            result.errors.append(f"duplicate_id:{s.id}")
        else:
            seen[s.id] = 1
        if s.text is None or str(s.text).strip() == "":
            result.errors.append(f"empty_text:{s.id}")
        if require_label and s.label is None:
            result.errors.append(f"missing_label:{s.id}")
        if s.label is not None and s.label not in valid_labels:
            result.errors.append(f"invalid_label:{s.id}={s.label}")
        if s.published_at is not None and s.published_at.tzinfo is None:
            result.errors.append(f"missing_timezone:{s.id}")
        elif s.published_at is None and require_label:
            pass  # published_at 可选
    return result


# --------------------------------------------------------------------------- #
# split（不重叠，seed 可复现）
# --------------------------------------------------------------------------- #
def split_dataset(
    samples: Sequence[Sample],
    train_frac: float = 0.7,
    dev_frac: float = 0.15,
    seed: int = 42,
    stratify_key: Optional[str] = None,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """划分 train/dev/test，互不重叠；支持按 label 分层。

    返回 (train, dev, test)。test 始终 >= 1 - train_frac - dev_frac。
    """
    if not (0 < train_frac < 1 and 0 <= dev_frac < 1 and train_frac + dev_frac < 1):
        raise ValueError("train_frac/dev_frac 需满足 0 < train < 1, 0 <= dev, train+dev < 1")
    items = list(samples)
    rng = random.Random(seed)
    if stratify_key == "label":
        if items and items[0].label is None:
            raise ValueError("stratify_key='label' 时所有样本必须有 label，不能静默退化为随机切分")
        missing_labels = [s.id for s in items if s.label is None]
        if missing_labels:
            raise ValueError(f"stratify_key='label' 时存在无 label 样本: {missing_labels[:5]}")
    if stratify_key == "label":
        by_label: Dict[Optional[int], List[Sample]] = {}
        for s in items:
            by_label.setdefault(s.label, []).append(s)
        train, dev, test = [], [], []
        for group in by_label.values():
            rng.shuffle(group)
            n = len(group)
            n_tr = int(n * train_frac)
            n_dev = int(n * dev_frac)
            train += group[:n_tr]
            dev += group[n_tr:n_tr + n_dev]
            test += group[n_tr + n_dev:]
        rng.shuffle(train)
        rng.shuffle(dev)
        rng.shuffle(test)
        return train, dev, test
    rng.shuffle(items)
    n = len(items)
    n_tr = int(n * train_frac)
    n_dev = int(n * dev_frac)
    return items[:n_tr], items[n_tr:n_tr + n_dev], items[n_tr + n_dev:]


def assert_no_overlap(train, dev, test) -> None:
    """train/dev/test 不重叠检查。"""
    tr = {s.id for s in train}
    dv = {s.id for s in dev}
    te = {s.id for s in test}
    assert tr.isdisjoint(dv) and tr.isdisjoint(te) and dv.isdisjoint(te), "split 重叠!"


# --------------------------------------------------------------------------- #
# DataLoader（动态 padding，保留样本 ID）
# --------------------------------------------------------------------------- #
def make_dataloader(
    samples: Sequence[Sample],
    tokenizer,
    batch_size: int = 8,
    max_length: int = 128,
    shuffle: bool = False,
    seed: int = 0,
    return_dict: bool = True,
):
    """构造 (input_dict, labels, ids) 批的迭代器。"""

    def _collate(batch: List[Sample]):
        texts = [s.text for s in batch]
        enc = tokenizer(
            texts, max_length=max_length, truncation=True, padding=True, return_tensors="pt",
        )
        labels = [s.label for s in batch]
        ids = [s.id for s in batch]
        return enc, labels, ids

    order = list(range(len(samples)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield _collate([samples[i] for i in idx])


def generate_synthetic_dataset(
    n: int = 64,
    seed: int = 42,
    include_metadata: bool = True,
) -> List[Sample]:
    """生成合成数据集（伪标签，仅验证管线，不构成任何表现结论）。"""
    rng = random.Random(seed)
    texts = [
        "市场今天表现平稳，投资者观望情绪浓厚。",
        "公司发布季度财报，营收同比增长明显。",
        "政策出台后，相关板块出现异动。",
        "分析师上调目标价，评级维持买入。",
        "风险事件发酵，市场波动加大。",
    ]
    sources = ["synthetic_a", "synthetic_b"]
    samples = []
    for i in range(n):
        label = rng.choice([0, 1])  # 伪标签：仅用于验证反向传播与梯度
        meta = {}
        if include_metadata:
            iso = (f"2024-0{rng.randint(1, 9)}-{rng.randint(1, 28):02d}"
                   f"T0{rng.randint(0, 9)}:{rng.randint(0, 59):02d}:00+08:00")
            dt, _ = parse_datetime_utc(iso)
            meta = {
                "entity_id": f"SYN-{rng.randint(1, 20)}",
                "published_at": dt,
                "source": rng.choice(sources),
            }
        samples.append(Sample(id=f"syn_{i}", text=rng.choice(texts),
                              label=label, **meta))
    return samples
