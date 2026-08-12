"""只训练 heads 的训练入口（P7 §9.3）。

未来冻结 backbone 时优先：
1. 一次前向缓存 12 层 pooled feature；
2. 针对每层特征训练独立 head；
3. optimizer 只接收 head 参数；
4. 每层采用相同 split、训练预算和超参数搜索空间。

当前只用合成数据证明：loss 能反向传播、head 参数变化、backbone 完全不变、
checkpoint 可保存重载、12 个头指标输出格式完整。**合成数据结果不提交为模型表现结论。**
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .layer_outputs import CacheVersionError, load_pooled_feature_cache, save_pooled_feature_cache


@dataclass
class TrainingSummary:
    head_type: str
    n_epochs: int
    per_layer_train_loss: List[float] = field(default_factory=list)
    per_layer_train_acc: List[float] = field(default_factory=list)
    backbone_unchanged: bool = True
    synthetic_only: bool = True


# --------------------------------------------------------------------------- #
# 特征缓存（一次前向）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_cached_features(
    model: nn.Module,
    dataloader,
    pooling: str,
    model_hash: str,
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[int], List[str]]:
    """一次前向缓存 12 层 pooled feature（不缓存整块 token hidden state）。

    返回 (features [N,12,768], labels, ids)。
    """
    feats, labels, ids = [], [], []
    for enc, batch_labels, batch_ids in dataloader:
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        feats.append(out.pooled_features.detach().cpu())   # [B,12,768]
        labels.extend(batch_labels)
        ids.extend(batch_ids)
    return torch.cat(feats, dim=0), labels, ids


def cache_features_from_model(
    model: nn.Module,
    dataloader,
    pooling: str,
    model_hash: str,
    cache_path: str,
    device: str = "cpu",
) -> None:
    """前向缓存并落盘（NPZ + 元数据版本保护）。"""
    feats, labels, ids = compute_cached_features(model, dataloader, pooling, model_hash, device)
    attn_lens = [len(str(i)) for i in ids]
    save_pooled_feature_cache(cache_path, feats, ids, attn_lens, pooling, model_hash,
                              extra={"labels": labels})
    return None


def load_cached_features(
    cache_path: str,
    expected_pooling: str,
    expected_model_hash: str,
) -> Tuple[torch.Tensor, List[int], List[str]]:
    """加载缓存并按版本校验；pooling/hash 不符抛 CacheVersionError。"""
    data = load_pooled_feature_cache(cache_path, expected_pooling=expected_pooling,
                                     expected_model_hash=expected_model_hash)
    feats = torch.tensor(data["pooled_features"], dtype=torch.float32)
    labels = data["meta"].get("labels")
    if labels is None:
        raise CacheVersionError("缓存缺少 labels 元数据")
    return feats, labels, data["text_ids"]


# --------------------------------------------------------------------------- #
# 只训练 head
# --------------------------------------------------------------------------- #
def train_heads_from_features(
    model: nn.Module,
    features: torch.Tensor,       # [N, 12, 768]
    labels: Sequence[int],        # [N]
    head_type: str = "copied_layer_heads",
    n_epochs: int = 3,
    lr: float = 0.01,
    batch_size: int = 16,
    device: str = "cpu",
) -> TrainingSummary:
    """只用缓存的 pooled features 训练指定 head；backbone 完全不参与。

    训练前记录 backbone 权重，训练后断言不变。返回每层训练 loss/acc。
    """
    if head_type not in ("copied_layer_heads", "random_layer_heads", "normalized_layer_heads"):
        raise ValueError(f"head_type={head_type} 不是可训练逐层头")
    module = getattr(model, head_type)
    features = features.to(device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    # 记录 backbone 权重指纹
    backbone_before = [p.detach().clone() for p in model.bert.parameters()]

    # optimizer 只接收该 head 的参数
    params = [p for p in module.parameters() if p.requires_grad]
    assert params, f"{head_type} 没有可训练参数"
    optimizer = torch.optim.AdamW(params, lr=lr)

    n = features.shape[0]
    n_layers = features.shape[1]
    losses = torch.zeros(n_layers, device=device)
    correct = torch.zeros(n_layers, device=device)
    seen = torch.zeros(n_layers, device=device)

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            feats = features[idx]                    # [B,12,768]
            tgt = labels_t[idx]                      # [B]
            logits = module(feats)                   # [B,12,2]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), tgt.repeat(n_layers))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)         # [B,12]
                for layer in range(n_layers):
                    losses[layer] += torch.nn.functional.cross_entropy(
                        logits[:, layer], tgt).item() * idx.numel()
                    correct[layer] += (pred[:, layer] == tgt).sum().item()
                    seen[layer] += idx.numel()

    # backbone 不变断言
    backbone_unchanged = all(
        torch.equal(before, after.detach())
        for before, after in zip(backbone_before, model.bert.parameters())
    )

    return TrainingSummary(
        head_type=head_type,
        n_epochs=n_epochs,
        per_layer_train_loss=[(losses[l] / seen[l]).item() for l in range(n_layers)],
        per_layer_train_acc=[(correct[l] / seen[l]).item() for l in range(n_layers)],
        backbone_unchanged=backbone_unchanged,
        synthetic_only=True,
    )


# --------------------------------------------------------------------------- #
# checkpoint 保存/重载
# --------------------------------------------------------------------------- #
def save_head_checkpoint(model: nn.Module, head_type: str, path: str) -> None:
    """保存指定 head 的 state_dict（safetensors 优先）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    state = {k: v.detach().clone().contiguous()
             for k, v in getattr(model, head_type).state_dict().items()}
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file(state, path)
    else:
        torch.save(state, path)


def load_head_checkpoint(model: nn.Module, head_type: str, path: str) -> None:
    """从 checkpoint 加载 head 权重到模型。"""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    getattr(model, head_type).load_state_dict(state, strict=True)
