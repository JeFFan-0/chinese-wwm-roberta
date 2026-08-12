"""P6 测试：修正后分析的统计正确性、padding 不变性与层编号规范。"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.analysis import (  # noqa: E402
    aggregate_layer,
    attention_comparison,
    hidden_state_comparison,
    load_pair,
    masked_activation_stats,
    per_layer_weight_metrics,
    tensor_metrics,
)

CKPT = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")
BASE_WEIGHTS = os.path.join(ROOT, "chinese-roberta-wwm-ext", "pytorch_model.bin")


# --------------------------------------------------------------------------- #
# 层级聚合可从张量级原始值精确复算
# --------------------------------------------------------------------------- #
def test_aggregate_layer_recomputable():
    a = torch.randn(768, 768)
    b = torch.randn(768, 768)
    m = tensor_metrics(a, b)
    agg = aggregate_layer([m])
    # 单张量时聚合公式应等于张量级原始值
    assert agg["delta_l2_layer"] == pytest.approx(m["delta_l2"])
    assert agg["base_l2_layer"] == pytest.approx(m["base_l2"])
    assert agg["relative_l2_layer"] == pytest.approx(m["relative_l2"])
    assert agg["max_abs_delta_layer"] == pytest.approx(m["max_abs_delta"])
    # 多张量：按公式复算
    ms = [tensor_metrics(torch.randn(768, 768), torch.randn(768, 768)) for _ in range(3)]
    agg2 = aggregate_layer(ms)
    assert agg2["delta_l2_layer"] == pytest.approx(
        sum(x["delta_l2"] ** 2 for x in ms) ** 0.5)
    assert agg2["base_l2_layer"] == pytest.approx(
        sum(x["base_l2"] ** 2 for x in ms) ** 0.5)
    mae_manual = sum(x["mean_abs_delta"] * x["numel"] for x in ms) / sum(x["numel"] for x in ms)
    assert agg2["mae_layer"] == pytest.approx(mae_manual)


# --------------------------------------------------------------------------- #
# 相同 state dict 自比较：delta=0，cos=1
# --------------------------------------------------------------------------- #
def test_per_layer_weight_self_compare(ckpt_path):
    ft, _ = load_pair(CKPT, BASE_WEIGHTS)
    tensor_rows, layer_rows = per_layer_weight_metrics(ft, ft)
    assert all(r["max_abs_delta"] == 0.0 for r in tensor_rows)
    assert all(r["cosine"] == pytest.approx(1.0) for r in tensor_rows)
    assert all(r["delta_l2_layer"] == 0.0 for r in layer_rows)
    assert all(r["relative_l2_layer"] == 0.0 for r in layer_rows)
    assert len(layer_rows) == 12  # encoder layers 0-11


def test_per_layer_weight_layer_numbering(ckpt_path):
    ft, base = load_pair(CKPT, BASE_WEIGHTS)
    tensor_rows, layer_rows = per_layer_weight_metrics(ft, base)
    assert [r["encoder_layer"] for r in layer_rows] == list(range(12))
    assert all(r["hidden_index"] == r["encoder_layer"] + 1 for r in layer_rows)


# --------------------------------------------------------------------------- #
# 真实 max abs delta（不再是 mean_abs_diff 断言）
# --------------------------------------------------------------------------- #
def test_max_abs_delta_ge_mean(ckpt_path):
    ft, base = load_pair(CKPT, BASE_WEIGHTS)
    tensor_rows, _ = per_layer_weight_metrics(ft, base)
    for r in tensor_rows:
        assert r["max_abs_delta"] >= r["mean_abs_delta"]


# --------------------------------------------------------------------------- #
# masked 激活统计：padding 不变性 + 层编号
# --------------------------------------------------------------------------- #
def test_masked_activation_padding_invariance():
    hidden = [torch.randn(2, 20, 768) for _ in range(13)]
    mask_short = torch.ones(2, 10, dtype=torch.long)  # 只有前 10 个 token
    # 构造更长 mask：右侧 padding，第 2 行全有效
    mask_long = torch.cat([mask_short, torch.zeros(2, 10, dtype=torch.long)], dim=1)
    # 需要 hidden 与 mask 长度一致；用 20 长度 mask 对比不同有效长度
    mask_a = torch.ones(2, 20, dtype=torch.long)
    mask_a[1, 15:] = 0
    mask_b = mask_a.clone()  # 加 padding 到 20 后右侧再加 10 个 pad
    hidden_b = [torch.cat([t, torch.zeros(2, 10, 768)], dim=1) for t in hidden]
    mask_b_long = torch.cat([mask_b, torch.zeros(2, 10, dtype=torch.long)], dim=1)

    stats_a = masked_activation_stats(hidden, mask_a)
    stats_b = masked_activation_stats(hidden_b, mask_b_long)
    for ra, rb in zip(stats_a, stats_b):
        assert ra["hidden_index"] == rb["hidden_index"]
        # 有效 token 统计应不受右侧额外 padding 影响
        assert ra["token_micro_mean_abs"] == pytest.approx(rb["token_micro_mean_abs"], rel=1e-3)
        assert ra["token_micro_max_abs"] == pytest.approx(rb["token_micro_max_abs"], rel=1e-3)


def test_activation_layer_numbering():
    hidden = [torch.randn(1, 8, 768) for _ in range(13)]
    mask = torch.ones(1, 8, dtype=torch.long)
    rows = masked_activation_stats(hidden, mask)
    assert rows[0]["encoder_layer"] is None          # hidden index 0 = embedding
    assert [r["encoder_layer"] for r in rows[1:]] == list(range(12))


# --------------------------------------------------------------------------- #
# 注意力 mask：只统计有效 query/key
# --------------------------------------------------------------------------- #
def test_attention_comparison_self():
    attn = [torch.randn(2, 12, 16, 16) for _ in range(12)]
    mask = torch.ones(2, 16, dtype=torch.long)
    rows = attention_comparison(attn, attn, mask)
    assert len(rows) == 12
    assert all(r["attn_cos_masked"] == pytest.approx(1.0) for r in rows)


def test_attention_comparison_layer_numbering():
    attn = [torch.randn(1, 12, 8, 8) for _ in range(12)]
    mask = torch.ones(1, 8, dtype=torch.long)
    rows = attention_comparison(attn, attn, mask)
    assert [r["encoder_layer"] for r in rows] == list(range(12))
    assert all(r["hidden_index"] == r["encoder_layer"] + 1 for r in rows)


# --------------------------------------------------------------------------- #
# hidden state 比较 micro/macro 双口径
# --------------------------------------------------------------------------- #
def test_hidden_state_comparison_self():
    hs = [torch.randn(2, 10, 768) for _ in range(13)]
    mask = torch.ones(2, 10, dtype=torch.long)
    rows = hidden_state_comparison(hs, hs, mask)
    assert all(r["cos_micro"] == pytest.approx(1.0) for r in rows)
    assert all(r["cos_macro"] == pytest.approx(1.0) for r in rows)
    assert all(r["l2_micro"] == pytest.approx(0.0) for r in rows)


# --------------------------------------------------------------------------- #
# load_pair 解包正确（修复 for 循环内无效赋值）
# --------------------------------------------------------------------------- #
def test_load_pair_returns_stripped_dicts():
    ft, base = load_pair(CKPT, BASE_WEIGHTS)
    assert "bert.encoder.layer.0.attention.self.query.weight" in ft
    assert "fc.weight" in ft
    assert "bert.embeddings.word_embeddings.weight" in base
    # 不再残留包装键
    assert "model_state_dict" not in ft
