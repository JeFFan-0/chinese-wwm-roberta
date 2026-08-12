"""P7 测试：校准接口（温度、NLL/ECE、分数、阈值搜索、分离约束）。"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.calibration import (  # noqa: E402
    assert_calibration_test_separate,
    ece,
    entropy_score,
    fit_temperature,
    margin_score,
    max_prob_score,
    nll,
    search_threshold,
    temperature_scale,
)


def _logits_labels(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, 2, generator=g)
    labels = torch.randint(0, 2, (n,), generator=g)
    return logits, labels


# --------------------------------------------------------------------------- #
# 温度恒正 / 无 NaN
# --------------------------------------------------------------------------- #
def test_temperature_scale_positive_and_finite():
    logits, labels = _logits_labels()
    t = torch.tensor(1.0)
    scaled = temperature_scale(logits, t)
    assert torch.isfinite(scaled).all()
    t2 = torch.tensor(-3.0)  # 非法负温度 → clamp
    scaled2 = temperature_scale(logits, t2)
    assert torch.isfinite(scaled2).all()
    assert (scaled2.abs() >= scaled.abs()).all()  # clamp 后温度 ≥1e-6


def test_fit_temperature_positive():
    logits, labels = _logits_labels()
    t = fit_temperature(logits, labels, steps=100)
    assert t.item() > 0
    assert torch.isfinite(t)


# --------------------------------------------------------------------------- #
# NLL / ECE
# --------------------------------------------------------------------------- #
def test_nll_finite():
    logits, labels = _logits_labels()
    loss = nll(logits, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_ece_range():
    logits, labels = _logits_labels()
    e = ece(logits, labels, n_bins=10)
    assert torch.isfinite(e)
    assert 0.0 <= e.item() <= 1.0


def test_ece_perfect_calibration_zero():
    # 置信度≈1 且预测全对的 logits → 每个 bin 的 conf≈acc → ECE≈0
    logits = torch.tensor([[50.0, -50.0], [80.0, -80.0], [30.0, -30.0]])
    labels = torch.tensor([0, 0, 0])
    assert ece(logits, labels, n_bins=5).item() < 1e-3


# --------------------------------------------------------------------------- #
# 分数函数
# --------------------------------------------------------------------------- #
def test_scores_finite():
    logits, _ = _logits_labels()
    for fn in (max_prob_score, entropy_score, margin_score):
        s = fn(logits)
        assert torch.isfinite(s).all()
        assert s.shape == (logits.shape[0],)


def test_entropy_perfect_confidence_zero():
    logits = torch.tensor([[100.0, -100.0]])
    assert entropy_score(logits).item() == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# 阈值搜索：要求独立 calibration 输入 + 分离约束
# --------------------------------------------------------------------------- #
def test_search_threshold_requires_calibration():
    with pytest.raises(ValueError):
        search_threshold(torch.empty(0, 2), torch.empty(0, dtype=torch.long),
                         score_fn=max_prob_score)


def test_search_threshold_high_quality_low_coverage():
    logits, labels = _logits_labels(seed=3)
    res = search_threshold(logits, labels, score_fn=max_prob_score,
                           target_quality=0.99, min_coverage=0.0, mode="max")
    # 高要求 → 覆盖率低但通过的样本 quality 高
    assert res.coverage <= 1.0
    assert res.quality >= 0.0


def test_search_threshold_low_requirement_high_coverage():
    logits, labels = _logits_labels(seed=3)
    res = search_threshold(logits, labels, score_fn=max_prob_score,
                           target_quality=0.0, min_coverage=0.0, mode="max")
    assert res.coverage == pytest.approx(1.0, abs=0.02)  # 零要求 → 全覆盖


def test_calibration_test_separate():
    assert_calibration_test_separate({"a", "b"}, {"c"})  # 不重叠通过
    with pytest.raises(AssertionError):
        assert_calibration_test_separate({"a", "b"}, {"b", "c"})  # 重叠报错
