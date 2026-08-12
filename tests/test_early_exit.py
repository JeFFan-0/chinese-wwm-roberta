"""P4 测试：真正逐层执行的 Early-Exit 引擎。

对应 TODO §11：test_final_exit_equivalence / test_forced_exit_stops_compute /
test_last_layer_fallback / test_active_batch_order。
全部在 CPU 运行。
"""
import pytest
import torch

from src.early_exit import EarlyExitEngine
from src.modeling import tokenize_texts

TEXT_A = "北京天气怎么样，明天会下雨吗？"
TEXT_B = "这个项目的核心目标是提升模型的每一层利用率，而不是只用最后一层。"
TEXT_C = "今天股市大涨，投资者情绪明显回暖。"


@pytest.fixture(scope="module")
def engine(heads_model):
    return EarlyExitEngine(heads_model.bert, heads_model, head_type="copied_layer_heads",
                           pooling="masked_mean")


@pytest.fixture(scope="module")
def bs1(engine, tokenizer):
    enc = tokenize_texts([TEXT_A], tokenizer)
    return {k: v for k, v in enc.items()}


@pytest.fixture(scope="module")
def bs2(engine, tokenizer):
    enc = tokenize_texts([TEXT_A, TEXT_B], tokenizer)
    return {k: v for k, v in enc.items()}


# --------------------------------------------------------------------------- #
# test_final_exit_equivalence：固定 layer 11 与完整前向一致
# --------------------------------------------------------------------------- #
def test_final_exit_equivalence(engine, bs2):
    result = engine.run_fixed(**bs2, exit_layer=11)
    baseline = engine.full_forward_baseline(**bs2)
    assert torch.allclose(result.logits, baseline, atol=1e-5, rtol=1e-5)
    assert result.exit_reason == "fixed_layer"
    assert result.exit_layer == 11


# --------------------------------------------------------------------------- #
# test_forced_exit_stops_compute：k+1..11 的 call counter 全为 0
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [2, 4, 6, 8])
def test_forced_exit_stops_compute(engine, bs2, k):
    result = engine.run_fixed(**bs2, exit_layer=k)
    counts = result.layer_call_counts
    assert sum(counts[: k + 1]) == k + 1
    assert sum(counts[k + 1:]) == 0, f"层 {k+1}..11 不应被调用: {counts}"
    assert counts[k] == 1


def test_executed_layer_count(engine, bs2):
    for k in (0, 3, 7, 11):
        result = engine.run_fixed(**bs2, exit_layer=k)
        assert result.executed_layer_count == k + 1


# --------------------------------------------------------------------------- #
# test_last_layer_fallback / 低阈值提前退出
# --------------------------------------------------------------------------- #
def test_last_layer_fallback_high_threshold(engine, bs1):
    result = engine.run_dynamic(**bs1, strategy="max_prob", threshold=1.0, candidate_layers=[2, 4, 6, 8, 10, 11])
    assert result.exit_reason == "fallback"
    assert result.exit_layer == 11
    assert result.executed_layer_count == 12
    assert result.layer_call_counts[11] == 1


def test_low_threshold_exits_first_candidate(engine, bs1):
    cands = [2, 4, 6, 8, 10, 11]
    result = engine.run_dynamic(**bs1, strategy="max_prob", threshold=-1000.0, candidate_layers=cands)
    assert result.exit_reason == "max_prob_threshold"
    assert result.exit_layer == cands[0]      # 首个候选层退出
    assert result.executed_layer_count == cands[0] + 1
    assert sum(result.layer_call_counts[cands[0] + 1:]) == 0


def test_margin_threshold_smoke(engine, bs1):
    result = engine.run_dynamic(**bs1, strategy="margin", threshold=-1e9, candidate_layers=[3, 6, 11])
    assert result.exit_reason == "margin_threshold"
    assert result.exit_layer == 3


def test_dynamic_bs1_only(engine, bs2):
    with pytest.raises(ValueError):
        engine.run_dynamic(**bs2, strategy="max_prob", threshold=0.9)


# --------------------------------------------------------------------------- #
# test_active_batch_order
# --------------------------------------------------------------------------- #
def test_active_batch_order(engine, tokenizer):
    """active-set 输出顺序与输入顺序一致，且与单条动态退出一致。"""
    enc = tokenize_texts([TEXT_A, TEXT_B, TEXT_C], tokenizer)
    result, reasons = engine.run_active_set(**enc, strategy="max_prob", threshold=-1000.0,
                                            candidate_layers=[2, 4, 6, 8, 10, 11])
    # 顺序一致：每行与对应单条动态退出对比
    for j, text in enumerate([TEXT_A, TEXT_B, TEXT_C]):
        single = tokenize_texts([text], tokenizer)
        ref = engine.run_dynamic(**single, strategy="max_prob", threshold=-1000.0,
                                 candidate_layers=[2, 4, 6, 8, 10, 11])
        assert torch.allclose(result.logits[j], ref.logits[0], atol=1e-4), f"text {j}"
        assert reasons[j] == ref.exit_reason
    assert torch.isfinite(result.probabilities).all()


def test_active_set_fallback_all(engine, tokenizer):
    enc = tokenize_texts([TEXT_A, TEXT_B], tokenizer)
    result, reasons = engine.run_active_set(**enc, strategy="max_prob", threshold=1.0,
                                            candidate_layers=[2, 4, 6, 8, 10, 11])
    assert all(r == "fallback" for r in reasons)
    assert torch.isfinite(result.probabilities).all()


# --------------------------------------------------------------------------- #
# 不同 batch 大小无 NaN/Inf
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 3, 5])
def test_no_nan_different_batch_sizes(engine, tokenizer, n):
    texts = [TEXT_A, TEXT_B, TEXT_C, TEXT_A, TEXT_B, TEXT_C][:n]
    enc = tokenize_texts(texts, tokenizer)
    result = engine.run_fixed(**enc, exit_layer=11)
    assert torch.isfinite(result.logits).all()
    assert torch.isfinite(result.probabilities).all()


# --------------------------------------------------------------------------- #
# 不同 head_type
# --------------------------------------------------------------------------- #
def test_shared_head_engine(heads_model, bs2):
    eng = EarlyExitEngine(heads_model.bert, heads_model, head_type="shared_frozen_head",
                          pooling="masked_mean")
    r = eng.run_fixed(**bs2, exit_layer=11)
    baseline = eng.full_forward_baseline(**bs2)
    assert torch.allclose(r.logits, baseline, atol=1e-5)


def test_original_final_head_only_fallback(heads_model, bs2):
    eng = EarlyExitEngine(heads_model.bert, heads_model, head_type="original_final_head",
                          pooling="masked_mean")
    r = eng.run_fixed(**bs2, exit_layer=11)
    assert r.exit_reason == "fixed_layer"
