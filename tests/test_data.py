"""P7 测试：数据协议（CSV/JSONL 加载、校验、split、动态 padding）。"""
import os
import sys
from datetime import timezone

import pytest
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import (  # noqa: E402
    assert_no_overlap,
    generate_synthetic_dataset,
    load_dataset,
    make_dataloader,
    parse_datetime_utc,
    split_dataset,
    validate_dataset,
)

LABEL_MAP = {"0": "unknown", "1": "unknown"}


@pytest.fixture
def synthetic():
    return generate_synthetic_dataset(n=40, seed=7, include_metadata=True)


def test_parse_datetime_utc_ok():
    dt, err = parse_datetime_utc("2024-01-02T03:04:05+08:00")
    assert err is None
    assert dt.tzinfo == timezone.utc
    # +08:00 03:04 应转换为 UTC 前一日 19:04
    assert dt == datetime(2024, 1, 1, 19, 4, 5, tzinfo=timezone.utc)


def test_parse_datetime_utc_fractional_seconds():
    dt, err = parse_datetime_utc("2024-01-02T03:04:05.123456+08:00")
    assert err is None
    assert dt.microsecond == 123456


def test_parse_datetime_utc_z():
    dt, err = parse_datetime_utc("2024-01-02T03:04:05Z")
    assert err is None
    assert dt.tzinfo == timezone.utc
    assert dt == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_parse_datetime_utc_invalid():
    dt, err = parse_datetime_utc("2024-01-02 03:04:05")  # 无时区
    assert err == "invalid_datetime_format"
    dt2, err2 = parse_datetime_utc("not-a-date")
    assert err2 == "invalid_datetime_format"


def test_validate_synthetic_ok(synthetic):
    vr = validate_dataset(synthetic, LABEL_MAP)
    assert vr.ok, vr.errors


def test_validate_detects_errors(synthetic):
    bad = [synthetic[0], synthetic[1], synthetic[2]]
    bad[0].id = synthetic[1].id        # 重复 ID
    bad[1].text = ""                    # 空文本
    bad[2].label = 7                    # 非法 label
    vr = validate_dataset(bad, LABEL_MAP)
    assert any("duplicate_id" in e for e in vr.errors)
    assert any("empty_text" in e for e in vr.errors)
    assert any("invalid_label" in e for e in vr.errors)


def test_load_csv_and_jsonl(tmp_path, synthetic):
    import csv
    import json
    csv_p = tmp_path / "d.csv"
    jsonl_p = tmp_path / "d.jsonl"
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label", "published_at", "source"])
        for s in synthetic[:5]:
            w.writerow([s.id, s.text, s.label,
                        s.published_at.isoformat() if s.published_at else "", s.source or ""])
    with open(jsonl_p, "w", encoding="utf-8") as f:
        for s in synthetic[:5]:
            f.write(json.dumps({"id": s.id, "text": s.text, "label": s.label,
                                "published_at": s.published_at.isoformat() if s.published_at else None,
                                "source": s.source}) + "\n")
    from_csv = load_dataset(str(csv_p), LABEL_MAP)
    from_jsonl = load_dataset(str(jsonl_p), LABEL_MAP)
    assert len(from_csv) == len(from_jsonl) == 5
    assert [s.id for s in from_csv] == [s.id for s in from_jsonl]


def test_split_no_overlap_and_reproducible(synthetic):
    tr1, dv1, te1 = split_dataset(synthetic, 0.7, 0.15, seed=42, stratify_key="label")
    tr2, dv2, te2 = split_dataset(synthetic, 0.7, 0.15, seed=42, stratify_key="label")
    assert [s.id for s in tr1] == [s.id for s in tr2]
    assert_no_overlap(tr1, dv1, te1)
    assert len(tr1) + len(dv1) + len(te1) == len(synthetic)


def test_split_invalid_frac(synthetic):
    with pytest.raises(ValueError):
        split_dataset(synthetic, 0.7, 0.4, seed=1)


def test_dataloader_dynamic_padding(synthetic, tokenizer):
    dl = list(make_dataloader(synthetic[:8], tokenizer, batch_size=4, shuffle=True, seed=0))
    assert len(dl) == 2
    for enc, labels, ids in dl:
        assert enc["input_ids"].shape[0] == 4
        assert len(labels) == 4 == len(ids)


def test_dataloader_preserves_ids(synthetic, tokenizer):
    dl = list(make_dataloader(synthetic[:4], tokenizer, batch_size=4, shuffle=False))
    enc, labels, ids = dl[0]
    assert ids == [s.id for s in synthetic[:4]]
