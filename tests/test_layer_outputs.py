"""P3 测试：逐层输出 schema、层编号、概率有效性、重读稳定与 batch 顺序不变性。

对应 TODO §11：test_layer_mapping / test_probability_validity /
test_forced_exit... 之外的 P3 验收项。
"""
import os

import pytest
import torch

from src.probes.layer_outputs import expected_row_count, layer_diagnostics, layer_head_rows
from src.models.modeling import tokenize_texts

TEXT_A = "北京天气怎么样，明天会下雨吗？"
TEXT_B = "这个项目的核心目标是提升模型的每一层利用率，而不是只用最后一层。"
TEXT_C = "今天股市大涨，投资者情绪明显回暖。"


def _run(model, tokenizer, texts, text_ids):
    enc = tokenize_texts(texts, tokenizer)
    with torch.no_grad():
        out = model(**enc)
    return layer_head_rows(out, text_ids, model.model_hash, model.pooling)


# --------------------------------------------------------------------------- #
# test_layer_mapping
# --------------------------------------------------------------------------- #
def test_layer_mapping(heads_model, tokenizer):
    df = _run(heads_model, tokenizer, [TEXT_A], ["t0"])
    assert (df["hidden_index"] == df["encoder_layer"] + 1).all()
    # 生产逐层头覆盖 encoder layer 0-11
    sub = df[df["head_type"] == "copied_layer_heads"]
    assert sorted(sub["encoder_layer"].unique()) == list(range(12))
    # 最终头 A 只导出 encoder layer 11（hidden_index 12）
    final = df[df["head_type"] == "original_final_head"]
    assert set(final["encoder_layer"]) == {11}
    assert set(final["hidden_index"]) == {12}


# --------------------------------------------------------------------------- #
# 行数、概率、字段
# --------------------------------------------------------------------------- #
def test_row_count(heads_model, tokenizer):
    df = _run(heads_model, tokenizer, [TEXT_A, TEXT_B], ["t0", "t1"])
    per_layer = ["shared_frozen_head", "copied_layer_heads", "random_layer_heads",
                 "normalized_layer_heads"]
    expected = expected_row_count(2, len(per_layer), include_final_head=True)
    assert len(df) == expected == 2 * (12 * 4 + 1)


def test_probability_validity(heads_model, tokenizer):
    df = _run(heads_model, tokenizer, [TEXT_A], ["t0"])
    p0 = df["class_0_prob"]
    p1 = df["class_1_prob"]
    assert p0.notna().all() and p1.notna().all()
    assert torch.isfinite(torch.tensor(df["class_0_logit"].values)).all()
    assert ((p0 + p1) - 1.0).abs().max() < 1e-5


def test_schema_columns(heads_model, tokenizer):
    from src.probes.layer_outputs import UNIFIED_COLUMNS
    df = _run(heads_model, tokenizer, [TEXT_A], ["t0"])
    assert list(df.columns) == UNIFIED_COLUMNS


def test_model_hash_recorded(heads_model, tokenizer):
    df = _run(heads_model, tokenizer, [TEXT_A], ["t0"])
    assert (df["model_hash"] == "test-hash").all()


# --------------------------------------------------------------------------- #
# 重读稳定（导出 CSV/Parquet 后字段与行数不变）
# --------------------------------------------------------------------------- #
def test_reexport_stable(heads_model, tokenizer, tmp_path):
    import pandas as pd
    df = _run(heads_model, tokenizer, [TEXT_A, TEXT_B], ["t0", "t1"])
    csv = tmp_path / "out.csv"
    parquet = tmp_path / "out.parquet"
    df.to_csv(csv, index=False)
    df.to_parquet(parquet, index=False)
    df_csv = pd.read_csv(csv)
    df_pq = pd.read_parquet(parquet)
    assert list(df_csv.columns) == list(df.columns)
    assert len(df_csv) == len(df)
    assert list(df_pq.columns) == list(df.columns)
    assert len(df_pq) == len(df)
    # 值一致（CSV 字符串转换后按近似比较）
    assert df_csv["class_0_logit"].max() == pytest.approx(df["class_0_logit"].max(), rel=1e-3)


# --------------------------------------------------------------------------- #
# batch 顺序变化不改变相同 text_id 结果
# --------------------------------------------------------------------------- #
def test_batch_order_invariance(heads_model, tokenizer):
    df_ab = _run(heads_model, tokenizer, [TEXT_A, TEXT_B], ["ta", "tb"])
    df_ba = _run(heads_model, tokenizer, [TEXT_B, TEXT_A], ["tb", "ta"])
    for tid in ("ta", "tb"):
        a = df_ab[df_ab["text_id"] == tid].sort_values(["head_type", "encoder_layer"])
        b = df_ba[df_ba["text_id"] == tid].sort_values(["head_type", "encoder_layer"])
        assert len(a) == len(b)
        diff = (a["class_0_logit"].values - b["class_0_logit"].values)
        assert abs(diff).max() < 1e-4, f"text_id={tid} batch 顺序改变结果"


# --------------------------------------------------------------------------- #
# 无标签诊断
# --------------------------------------------------------------------------- #
def test_layer_diagnostics(heads_model, tokenizer):
    df = _run(heads_model, tokenizer, [TEXT_A, TEXT_B], ["t0", "t1"])
    diag = layer_diagnostics(df)
    assert not diag.empty
    # 每个样本产生 11 个相邻层对
    assert len(diag) == 2 * 11
    assert set(diag["text_id"]) == {"t0", "t1"}
    assert diag["logit_l2_change"].notna().all()
    assert diag["prob_change"].notna().all()


def test_diagnostics_only_uses_shared_and_final(heads_model, tokenizer):
    df = _run(heads_model, tokenizer, [TEXT_A], ["t0"])
    diag = layer_diagnostics(df)
    # 只基于 shared_frozen_head 的相邻层 + 最终层一致率
    assert not diag.empty
