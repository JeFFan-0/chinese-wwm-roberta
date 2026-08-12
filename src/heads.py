"""逐层模型头（P2）。backbone 恒冻结。

Head A ``original_final_head``：原始最终头，只读副本，作用于最后层 -> [B, 2]
Head B ``shared_frozen_head``：同一 fc 依次作用于 12 层 -> [B, 12, 2]
Head C ``copied_layer_heads``：12 个复制头（初始 = 原 fc），未来只训练这些头
Head D ``random_layer_heads``：12 个随机头（固定 seed 可复现）
Head E ``normalized_layer_heads``：LayerNorm -> Dropout -> Linear（非默认）

层编号规范：encoder layer 0-11；hidden index = encoder_layer + 1。
embedding 头（hidden index 0）仅诊断，不进入 Early-Exit，也不会被误标成 encoder layer 0。
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .modeling import load_backbone, load_state_dict_safe

N_ENCODER_LAYERS = 12
HIDDEN_SIZE = 768
N_CLASSES = 2

_HEAD_TYPES = ("original_final_head", "shared_frozen_head", "copied_layer_heads",
               "random_layer_heads", "normalized_layer_heads")
_PRODUCTION_LAYER_HEADS = ("shared_frozen_head", "copied_layer_heads",
                           "random_layer_heads", "normalized_layer_heads")


# --------------------------------------------------------------------------- #
# Head A：原始最终头（只读）
# --------------------------------------------------------------------------- #
class OriginalFinalHead(nn.Module):
    """从 checkpoint 加载的原 fc，保持只读副本，仅作用于最后一个 encoder layer。"""

    def __init__(self, fc: nn.Linear):
        super().__init__()
        self.fc = fc
        self.fc.requires_grad_(False)
        # 只读参考副本，用于"参数逐元素等于 checkpoint fc"验证
        self.register_buffer("ref_weight", fc.weight.detach().clone())
        self.register_buffer("ref_bias", fc.bias.detach().clone())

    def forward(self, pooled_final: torch.Tensor) -> torch.Tensor:
        return self.fc(pooled_final)  # [B, 768] -> [B, 2]


# --------------------------------------------------------------------------- #
# Head B：共享冻结头
# --------------------------------------------------------------------------- #
class SharedFrozenHead(nn.Module):
    """同一个冻结 fc 依次作用于 12 层 pooled feature。

    衡量浅层特征与最终层分类平面的兼容程度，不衡量浅层情绪信息的上限。
    """

    def __init__(self, fc: nn.Linear):
        super().__init__()
        self.head = fc
        self.head.requires_grad_(False)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: [B, 12, 768] -> [B, 12, 2]"""
        return self.head(pooled)


# --------------------------------------------------------------------------- #
# Head C：12 个复制线性头
# --------------------------------------------------------------------------- #
class CopiedLayerHeads(nn.Module):
    """为 encoder layer 0-11 各建独立 Linear(768, 2)，初始参数均复制自原 fc。

    当前只创建/序列化/前向；数据到位后只优化这些 heads，不修改 backbone。
    """

    def __init__(self, fc: nn.Linear, n_layers: int = N_ENCODER_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.heads = nn.ModuleList([nn.Linear(HIDDEN_SIZE, N_CLASSES) for _ in range(n_layers)])
        with torch.no_grad():
            for h in self.heads:
                h.weight.copy_(fc.weight)
                h.bias.copy_(fc.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        outs = [self.heads[i](pooled[:, i]) for i in range(self.n_layers)]
        return torch.stack(outs, dim=1)  # [B, 12, 2]


# --------------------------------------------------------------------------- #
# Head D：12 个随机对照头
# --------------------------------------------------------------------------- #
class RandomLayerHeads(nn.Module):
    """相同结构，固定 seed 随机初始化；保存 seed 与配置。未训练前不得解释输出语义。"""

    def __init__(self, seed: int = 42, n_layers: int = N_ENCODER_LAYERS,
                 in_features: int = HIDDEN_SIZE, out_features: int = N_CLASSES):
        super().__init__()
        self.seed = seed
        self.n_layers = n_layers
        # 在固定 seed 的 RNG 状态下构造，保证可精确复现
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            self.heads = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(n_layers)])
        finally:
            torch.random.set_rng_state(rng_state)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        outs = [self.heads[i](pooled[:, i]) for i in range(self.n_layers)]
        return torch.stack(outs, dim=1)  # [B, 12, 2]


# --------------------------------------------------------------------------- #
# Head E：轻量增强候选（LayerNorm -> Dropout -> Linear）
# --------------------------------------------------------------------------- #
class NormalizedLayerHeads(nn.Module):
    """LayerNorm(768) -> Dropout(p) -> Linear(768, 2)，最终线性层复制自原 fc。

    仅实现，不作为默认头；用于未来验证不同层尺度差异是否影响线性头。
    """

    def __init__(self, fc: nn.Linear, dropout: float = 0.1, n_layers: int = N_ENCODER_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.heads = nn.ModuleList()
        for _ in range(n_layers):
            seq = nn.Sequential(
                nn.LayerNorm(HIDDEN_SIZE),
                nn.Dropout(dropout),
                nn.Linear(HIDDEN_SIZE, N_CLASSES),
            )
            with torch.no_grad():
                seq[2].weight.copy_(fc.weight)
                seq[2].bias.copy_(fc.bias)
            self.heads.append(seq)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # 推理用 eval 模式，Dropout 无影响；训练时经 synthetic pipeline 控制
        outs = [self.heads[i](pooled[:, i]) for i in range(self.n_layers)]
        return torch.stack(outs, dim=1)  # [B, 12, 2]


# --------------------------------------------------------------------------- #
# 校准参数占位符
# --------------------------------------------------------------------------- #
class ExitCalibration(nn.Module):
    """每出口正值温度参数 T_l，初始 1.0。当前不拟合、不保存推荐阈值。"""

    def __init__(self, n_layers: int = N_ENCODER_LAYERS):
        super().__init__()
        self.register_buffer("temperature", torch.ones(n_layers))
        self.temperature_fit = False

    def scale_logits(self, logits: torch.Tensor, layer: int) -> torch.Tensor:
        """temperature scaling: logits / T_l。温度恒正（初值 1）。"""
        t = self.temperature[layer].clamp_min(1e-6)
        return logits / t


# --------------------------------------------------------------------------- #
# 输出容器
# --------------------------------------------------------------------------- #
@dataclass
class LayerHeadOutput:
    results: Dict[str, torch.Tensor]        # head_type -> [B,2] 或 [B,12,2]
    pooled_features: torch.Tensor           # [B, 12, 768]（encoder layers 0-11）
    hidden_states: Tuple[torch.Tensor, ...] # 13 个，index 0 = embedding

    def per_layer(self, head_type: str) -> torch.Tensor:
        """按 head type 返回 [B, 12, 2]（对 [B,2] 的最终头广播到 12 层则抛错）。"""
        t = self.results[head_type]
        if t.dim() == 3:
            return t
        raise ValueError(f"{head_type} 不是逐层头（shape={tuple(t.shape)}）")


# --------------------------------------------------------------------------- #
# 逐层头模型
# --------------------------------------------------------------------------- #
class LayerHeadsModel(nn.Module):
    """backbone（冻结）+ 各类逐层模型头 + 校准占位符。"""

    def __init__(
        self,
        bert: nn.Module,
        fc: nn.Linear,
        heads_enabled: Iterable[str],
        pooling: str = "masked_mean",
        random_seed: int = 42,
        normalized_dropout: float = 0.1,
        pooling_confirmed: bool = False,
        model_hash: str = "unknown",
    ):
        super().__init__()
        self.bert = bert
        self.bert.requires_grad_(False)   # backbone 恒冻结
        self.pooling = pooling
        self.pooling_confirmed = pooling_confirmed
        self.model_hash = model_hash
        self.heads_enabled = list(heads_enabled)
        self.random_seed = random_seed

        if "original_final_head" in self.heads_enabled:
            self.original_final_head = OriginalFinalHead(copy.deepcopy(fc))
        if "shared_frozen_head" in self.heads_enabled:
            self.shared_frozen_head = SharedFrozenHead(copy.deepcopy(fc))
        if "copied_layer_heads" in self.heads_enabled:
            self.copied_layer_heads = CopiedLayerHeads(fc)
        if "random_layer_heads" in self.heads_enabled:
            self.random_layer_heads = RandomLayerHeads(seed=random_seed)
        if "normalized_layer_heads" in self.heads_enabled:
            self.normalized_layer_heads = NormalizedLayerHeads(fc, dropout=normalized_dropout)

        self.calibration = ExitCalibration(n_layers=N_ENCODER_LAYERS)

    # ------------------------------------------------------------------ #
    def _pooled_per_layer(
        self, hidden_states: Tuple[torch.Tensor, ...], attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """对 encoder layers 0-11（hidden indices 1-12）做相同 pooling -> [B, 12, 768]。

        embedding 输出（hidden index 0）不进入生产接头。
        """
        from .pooling import apply_pooling

        pooled = torch.stack(
            [
                apply_pooling(self.pooling, hidden_states[i], attention_mask=attention_mask,
                              pooler=self.bert.pooler)
                for i in range(1, N_ENCODER_LAYERS + 1)
            ],
            dim=1,
        )
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> LayerHeadOutput:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hs = out.hidden_states
        pooled = self._pooled_per_layer(hs, attention_mask)

        results: Dict[str, torch.Tensor] = {}
        if hasattr(self, "original_final_head"):
            results["original_final_head"] = self.original_final_head(pooled[:, -1])
        if hasattr(self, "shared_frozen_head"):
            results["shared_frozen_head"] = self.shared_frozen_head(pooled)
        if hasattr(self, "copied_layer_heads"):
            results["copied_layer_heads"] = self.copied_layer_heads(pooled)
        if hasattr(self, "random_layer_heads"):
            results["random_layer_heads"] = self.random_layer_heads(pooled)
        if hasattr(self, "normalized_layer_heads"):
            results["normalized_layer_heads"] = self.normalized_layer_heads(pooled)
        return LayerHeadOutput(results=results, pooled_features=pooled, hidden_states=hs)

    # ------------------------------------------------------------------ #
    def trainable_head_parameters(self) -> List[nn.Parameter]:
        """只返回可训练 head 的参数（backbone 与只读头一律不在内）。"""
        params: List[nn.Parameter] = []
        for name in self.heads_enabled:
            if name == "original_final_head" or name == "shared_frozen_head":
                continue  # 只读/冻结头
            module = getattr(self, name)
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    def head_parameter_summary(self) -> Dict[str, Dict[str, int]]:
        """记录每个 head 的参数量与额外存储开销。"""
        summary: Dict[str, Dict[str, int]] = {}
        for name in self.heads_enabled:
            module = getattr(self, name)
            n_params = sum(p.numel() for p in module.parameters())
            n_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            summary[name] = {"param_count": n_params, "storage_bytes": n_bytes}
        return summary

    def verify_backbone_frozen(self) -> bool:
        """backbone 所有参数 requires_grad=False。"""
        return all(not p.requires_grad for p in self.bert.parameters())


# --------------------------------------------------------------------------- #
# 构造器
# --------------------------------------------------------------------------- #
def build_layer_heads_model(
    base_model_dir: str,
    checkpoint_path: str,
    heads_enabled: Optional[Iterable[str]] = None,
    pooling: str = "masked_mean",
    pooling_confirmed: bool = False,
    random_seed: int = 42,
    normalized_dropout: float = 0.1,
    model_hash: Optional[str] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LayerHeadsModel:
    """加载 checkpoint 的 backbone + fc，装配逐层头模型。

    heads_enabled 缺省为全部生产头（B/C/D/E）。
    """
    if heads_enabled is None:
        heads_enabled = list(_PRODUCTION_LAYER_HEADS)

    bert = load_backbone(base_model_dir)
    state = load_state_dict_safe(checkpoint_path, map_location="cpu")
    backbone = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    bert.load_state_dict(backbone, strict=True)

    if "fc.weight" not in state or "fc.bias" not in state:
        raise KeyError("checkpoint 缺少 fc.weight / fc.bias")
    fc = nn.Linear(HIDDEN_SIZE, N_CLASSES)
    with torch.no_grad():
        fc.weight.copy_(state["fc.weight"])
        fc.bias.copy_(state["fc.bias"])

    if model_hash is None:
        from .modeling import checkpoint_hash
        model_hash = checkpoint_hash(checkpoint_path)

    model = LayerHeadsModel(
        bert, fc, heads_enabled=list(heads_enabled), pooling=pooling,
        pooling_confirmed=pooling_confirmed, random_seed=random_seed,
        normalized_dropout=normalized_dropout, model_hash=model_hash,
    )
    return model.to(device=device, dtype=dtype).eval()
