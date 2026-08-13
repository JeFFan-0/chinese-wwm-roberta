"""校准与阈值搜索接口（P7 §9.4）。

提前实现函数签名：
- 每个出口温度缩放（温度恒正，初始 1）；
- NLL 与 ECE；
- 最大概率 / 熵 / margin / patience 分数；
- 在质量下降约束下搜索阈值；
- dev/calibration 与 test 严格分离。

当前只用合成 logits 做数值与控制流测试，**不输出推荐阈值**。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def temperature_scale(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """logits / T。温度必须恒正（clamp 到 1e-6 下限）。"""
    t = temperature.clamp_min(1e-6)
    return logits / t


def nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """平均负对数似然（标量）。"""
    return F.cross_entropy(logits, targets)


def ece(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> torch.Tensor:
    """Expected Calibration Error（标量）。输入 logits [N,C]，targets [N]。"""
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    acc = (pred == targets).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    total = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        # 每个 bin 用 (lo, hi]，首个 bin 含左端点，避免相邻 bin 在边界上重复计数
        if b == 0:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf > lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = acc[in_bin].mean()
        bin_conf = conf[in_bin].mean()
        total += (bin_conf - bin_acc).abs() * in_bin.float().mean()
    return total


# --------------------------------------------------------------------------- #
# 分数函数
# --------------------------------------------------------------------------- #
def max_prob_score(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1).max(dim=-1).values


def entropy_score(logits: torch.Tensor, base: float = 2.0) -> torch.Tensor:
    """Shannon 熵，单位可任意底数（默认 bit）。

    自然对数熵 H_nats 除以 ln(base) 得到 base 进制下的熵。
    """
    import math
    p = torch.softmax(logits, dim=-1).clamp_min(1e-12)
    h_nats = -(p * p.log()).sum(dim=-1)
    return h_nats / math.log(base)


def margin_score(logits: torch.Tensor) -> torch.Tensor:
    """二分类 logit 差绝对值。"""
    return (logits[:, 1] - logits[:, 0]).abs()


def patience_score(logits_history: Sequence[torch.Tensor], patience: int = 2) -> torch.Tensor:
    """连续 patience 次预测一致则满足退出（二元 0/1，按样本）。输入为逐层 logits 列表。"""
    batch = logits_history[-1].shape[0]
    if len(logits_history) <= patience:
        return torch.zeros(batch, dtype=torch.long, device=logits_history[-1].device)
    preds = torch.stack([l.argmax(dim=-1) for l in logits_history[-patience:]])
    consistent = (preds == preds[0]).all(dim=0)
    return consistent.long()


# --------------------------------------------------------------------------- #
# 阈值搜索（要求独立 calibration 输入）
# --------------------------------------------------------------------------- #
@dataclass
class ThresholdResult:
    threshold: float
    coverage: float        # 通过阈值的样本比例
    quality: float         # 通过样本的准确率
    calibrated: bool


def search_threshold(
    cal_logits: torch.Tensor,
    cal_targets: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    target_quality: float = 0.9,
    min_coverage: float = 0.0,
    mode: str = "max",           # max: 分数越大越可信；min: 越小越可信（如熵）
    grid: Optional[Sequence[float]] = None,
    metric: str = "accuracy",
) -> ThresholdResult:
    """在**独立 calibration 输入**上搜索分数阈值。

    约束：仅允许通过样本的 quality（如 accuracy）不低于 target_quality，
    在满足约束的阈值里选覆盖最大者。**不得**用 test 集做此搜索。
    返回的对象始终标记 calibrated=True（意味着这是 calibration 流程产物，
    不是推荐给生产的正式阈值——生产阈值还需等真实数据 + test 一次评估）。
    """
    cal_targets = cal_targets.long()
    if score_fn is None or cal_logits.shape[0] == 0:
        raise ValueError("search_threshold 需要非空 calibration 输入")
    scores = score_fn(cal_logits)                       # [N]
    if grid is None:
        grid = torch.linspace(scores.min().item(), scores.max().item(), 100).tolist()

    preds = cal_logits.argmax(dim=-1)
    acc = (preds == cal_targets).float()

    best: Optional[ThresholdResult] = None
    for thr in grid:
        if mode == "max":
            passed = scores >= thr
        else:
            passed = scores <= thr
        if passed.sum() == 0:
            continue
        quality = acc[passed].mean().item()
        coverage = passed.float().mean().item()
        if quality >= target_quality and coverage >= min_coverage:
            if best is None or coverage > best.coverage:
                best = ThresholdResult(threshold=float(thr), coverage=coverage,
                                       quality=quality, calibrated=True)
    if best is None:
        best = ThresholdResult(threshold=float("nan"), coverage=0.0,
                               quality=0.0, calibrated=True)
    return best


def assert_calibration_test_separate(cal_ids: set, test_ids: set) -> None:
    """calibration 与 test 严格分离检查。"""
    overlap = cal_ids & test_ids
    assert not overlap, f"calibration 与 test 重叠样本: {sorted(overlap)[:5]}"


def fit_temperature(
    cal_logits: torch.Tensor,
    cal_targets: torch.Tensor,
    init: float = 1.0,
    lr: float = 0.01,
    steps: int = 500,
) -> torch.Tensor:
    """用 NLL 拟合单一温度（标量）。温度恒正。"""
    t = torch.tensor(init, requires_grad=True, dtype=cal_logits.dtype)
    opt = torch.optim.SGD([t], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = nll(temperature_scale(cal_logits, t), cal_targets.long())
        loss.backward()
        opt.step()
        with torch.no_grad():
            t.clamp_min_(1e-6)
    return t.detach().clone()
