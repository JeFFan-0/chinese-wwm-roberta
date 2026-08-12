"""修正后的逐层对比分析模块（P6）。

修复 check.ipynb / compare_layers.py 的可信度问题：
1. state dict 解包显式返回赋值（不再 for 循环内无效赋值）；
2. 层级聚合按参数计算（delta_l2_layer = sqrt(Σ||ΔW_p||²) 等），不简单平均 rel_l2；
3. 计算**真实** max abs delta，不再凭 mean_abs_diff 断言"最大绝对误差"；
4. 层编号规范：hidden index 0 = embedding，encoder layer 0-11；
5. 激活/注意力统计只对 attention_mask==1，且提供 token-micro 与 sentence-macro 平均；
6. 结论措辞：用"累计表示差异"，不推断单层改动，不报 ReLU 死神经元（配置为 GELU），
   无标签时不报层任务价值。

本模块不产生任何"某一层最好/含多少情绪信息"的结论。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .checkpoint import load_state_dict_safe, strip_prefix, unwrap_state_dict
from .modeling import load_backbone

COMPONENTS = [
    "attention.self.query.weight", "attention.self.key.weight",
    "attention.self.value.weight", "attention.output.dense.weight",
    "intermediate.dense.weight", "output.dense.weight",
]
N_ENCODER_LAYERS = 12


# --------------------------------------------------------------------------- #
# 加载与对齐
# --------------------------------------------------------------------------- #
def load_pair(checkpoint_path: str, base_weights_path: str):
    """加载 ckpt 与底座 state dict，正确解包/去前缀，返回 (ft, base)。"""
    ft_raw = load_state_dict_safe(checkpoint_path, map_location="cpu")
    base_raw = load_state_dict_safe(base_weights_path, map_location="cpu")
    ft = strip_prefix(unwrap_state_dict(ft_raw))
    base = strip_prefix(unwrap_state_dict(base_raw))
    return ft, base


# --------------------------------------------------------------------------- #
# 张量级指标
# --------------------------------------------------------------------------- #
def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """对单个张量计算 numel/base L2/delta L2/rel L2/mae/真实 max abs delta/cosine。"""
    ta = a.flatten().double()
    tb = b.flatten().double()
    delta = ta - tb
    base_l2 = tb.norm().item()
    delta_l2 = delta.norm().item()
    denom = (ta.norm().item() * tb.norm().item())
    cos = float((ta @ tb).item() / denom) if denom > 0 else 1.0
    return {
        "numel": ta.numel(),
        "base_l2": base_l2,
        "delta_l2": delta_l2,
        "relative_l2": (delta_l2 / base_l2) if base_l2 > 0 else float("nan"),
        "mean_abs_delta": delta.abs().mean().item(),
        "max_abs_delta": delta.abs().max().item(),
        "cosine": cos,
    }


# --------------------------------------------------------------------------- #
# 层级聚合（按参数计算，非简单平均）
# --------------------------------------------------------------------------- #
def aggregate_layer(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """按 TODO §8.1 公式聚合一层内的所有参数张量。

    delta_l2_layer = sqrt(Σ ||ΔW_p||²)
    base_l2_layer  = sqrt(Σ ||W_base,p||²)
    rel_l2_layer   = delta_l2_layer / base_l2_layer
    mae_layer      = Σ ||ΔW_p||₁ / Σ numel(p)
    max_layer      = max |ΔW_p|
    mean_cos       = 各张量 cosine 的均值（仅作参考，不用于结论）
    """
    delta_sq = sum(m["delta_l2"] ** 2 for m in metrics)
    base_sq = sum(m["base_l2"] ** 2 for m in metrics)
    mae_num = sum(m["mean_abs_delta"] * m["numel"] for m in metrics)
    numel = sum(m["numel"] for m in metrics)
    return {
        "delta_l2_layer": delta_sq ** 0.5,
        "base_l2_layer": base_sq ** 0.5,
        "relative_l2_layer": (delta_sq ** 0.5) / (base_sq ** 0.5) if base_sq > 0 else float("nan"),
        "mae_layer": mae_num / numel if numel else float("nan"),
        "max_abs_delta_layer": max(m["max_abs_delta"] for m in metrics),
        "mean_cosine": sum(m["cosine"] for m in metrics) / len(metrics) if metrics else 1.0,
    }


def per_layer_weight_metrics(
    ft: Dict[str, torch.Tensor],
    base: Dict[str, torch.Tensor],
    components: Sequence[str] = COMPONENTS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 (张量级行, 层级聚合行)，均含 encoder_layer 0-11（hidden_index = layer+1）。"""
    tensor_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    for layer in range(N_ENCODER_LAYERS):
        layer_metrics: List[Dict[str, float]] = []
        for comp in components:
            key = f"bert.encoder.layer.{layer}.{comp}"
            if key not in ft or key not in base:
                continue
            m = tensor_metrics(ft[key], base[key])
            tensor_rows.append({
                "encoder_layer": layer, "hidden_index": layer + 1,
                "component": comp, **m,
            })
            layer_metrics.append(m)
        if layer_metrics:
            agg = aggregate_layer(layer_metrics)
            layer_rows.append({"encoder_layer": layer, "hidden_index": layer + 1, **agg})
    return tensor_rows, layer_rows


# --------------------------------------------------------------------------- #
# 激活统计（masked，只对 attention_mask==1）
# --------------------------------------------------------------------------- #
def masked_activation_stats(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    cls_index: int = 0,
) -> List[Dict[str, Any]]:
    """对每个 hidden index 计算 masked 激活统计。

    token-micro：所有有效 token 一起统计；
    sentence-macro：先逐句统计再平均。
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states[0].dtype)  # [B,S,1]
    rows = []
    for hidx, t in enumerate(hidden_states):
        valid = (t * mask)  # [B,S,H]，pad 位置置 0
        # token-micro：全部有效 token
        micro = valid[attention_mask == 1]
        # sentence-macro：逐句对有效 token 求均值再跨句平均
        summed = valid.sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        sent_mean = (summed / denom)  # [B,H]
        macro_mean = sent_mean.mean(dim=0)
        rows.append({
            "hidden_index": hidx,
            "encoder_layer": hidx - 1 if hidx >= 1 else None,  # hidden 0 = embedding
            "cls_norm": float(t[:, cls_index].norm(dim=-1).mean()),
            "token_micro_mean_abs": float(micro.abs().mean()) if micro.numel() else float("nan"),
            "token_micro_max_abs": float(micro.abs().max()) if micro.numel() else float("nan"),
            "sentence_macro_mean_abs": float(macro_mean.abs().mean()),
        })
    return rows


def hidden_state_comparison(
    hidden_ft: Sequence[torch.Tensor],
    hidden_base: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
) -> List[Dict[str, Any]]:
    """逐层（hidden index）的 ft vs base 激活比较，只对 attention_mask==1。

    同时给出 token-micro 与 sentence-macro 的 cosine / mean_l2_dist。
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_ft[0].dtype)
    rows = []
    for hidx, (t, b) in enumerate(zip(hidden_ft, hidden_base)):
        vt, vb = t * mask, b * mask
        # micro：按 token
        ft_tok = vt[attention_mask == 1].double()
        base_tok = vb[attention_mask == 1].double()
        cos_micro = float(torch.nn.functional.cosine_similarity(ft_tok, base_tok, dim=-1).mean())
        l2_micro = float((ft_tok - base_tok).norm(dim=-1).mean())
        # macro：按句
        denom = mask.sum(dim=1).clamp_min(1.0)
        s_ft = (vt.sum(dim=1) / denom).double()
        s_base = (vb.sum(dim=1) / denom).double()
        cos_macro = float(torch.nn.functional.cosine_similarity(s_ft, s_base, dim=-1).mean())
        l2_macro = float((s_ft - s_base).norm(dim=-1).mean())
        rows.append({
            "hidden_index": hidx,
            "encoder_layer": hidx - 1 if hidx >= 1 else None,
            "cos_micro": cos_micro, "l2_micro": l2_micro,
            "cos_macro": cos_macro, "l2_macro": l2_macro,
        })
    return rows


# --------------------------------------------------------------------------- #
# 注意力比较（mask 无效 query/key）
# --------------------------------------------------------------------------- #
def attention_comparison(
    attn_ft: Sequence[torch.Tensor],
    attn_base: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
) -> List[Dict[str, Any]]:
    """对每个 encoder layer 的 attention 计算 masked cosine（query 与 key 均有效）。"""
    rows = []
    q_valid = attention_mask[:, None, None, :].bool()   # [B,1,1,S] key 有效
    k_valid = attention_mask[:, None, :, None].bool()   # [B,1,S,1] query 有效
    valid = (q_valid & k_valid)                          # [B,1,S,S]，广播到 head 维
    for layer, (af, ab) in enumerate(zip(attn_ft, attn_base)):
        mask_full = valid.expand_as(af)                  # [B,H,S,S]
        va, vb = af[mask_full].double(), ab[mask_full].double()
        cos = float(torch.nn.functional.cosine_similarity(va, vb, dim=-1).mean())
        rows.append({"encoder_layer": layer, "hidden_index": layer + 1, "attn_cos_masked": cos})
    return rows


# --------------------------------------------------------------------------- #
# 局部层替换对比（§8.5）：同一输入分别过 base/finetuned 对应层
# --------------------------------------------------------------------------- #
def build_layer_models(base_dir: str, checkpoint_path: str, device: str = "cpu"):
    """构建 finetuned 与 base 两个 BertModel（共享同一结构），并加载权重。"""
    ft = load_backbone(base_dir)
    ft_state = strip_prefix(unwrap_state_dict(load_state_dict_safe(checkpoint_path)))
    ft.load_state_dict({k[len("bert."):]: v for k, v in ft_state.items() if k.startswith("bert.")},
                       strict=True)
    base = load_backbone(base_dir)  # 从底座 safetensors 加载
    ft.to(device).eval()
    base.to(device).eval()
    return ft, base


def local_layer_comparison(
    ft: nn.Module,
    base: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    """对每个 encoder layer：同一 hidden 输入分别过 ft/base 对应层，比较输出。

    隔离了累计输入漂移，衡量的是"该层自身权重差异导致的输出差异"（局部）。
    仅用于自选文本上的 smoke test，不形成总体结论。
    """
    from transformers.masking_utils import create_bidirectional_mask

    with torch.no_grad():
        embeds = base.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        mask = create_bidirectional_mask(base.config, embeds, attention_mask)
        h_common = embeds
        rows = []
        for layer in range(N_ENCODER_LAYERS):
            # 同一个 h_common 分别过 ft/base 的对应层，输出差异仅来自该层权重
            h_ft = ft.encoder.layer[layer](h_common, attention_mask=mask)
            h_base = base.encoder.layer[layer](h_common, attention_mask=mask)
            vft = h_ft[attention_mask == 1].double()
            vbase = h_base[attention_mask == 1].double()
            cos = float(torch.nn.functional.cosine_similarity(vft, vbase, dim=-1).mean())
            l2 = float((vft - vbase).norm(dim=-1).mean())
            rows.append({"encoder_layer": layer, "hidden_index": layer + 1,
                         "local_cos": cos, "local_mean_l2": l2})
            h_common = h_base  # 公共状态沿 base 路径推进
        return rows
