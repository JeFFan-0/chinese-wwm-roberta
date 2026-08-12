"""逐层输出、导出与表征缓存（P3）。

统一输出 schema（§5.1）对每条文本、每个 encoder layer、每个 head type 生成一行：
    text_id, hidden_index, encoder_layer, head_type, pooling,
    class_0_logit, class_1_logit, class_0_prob, class_1_prob,
    logit_margin_1_minus_0, max_probability, entropy, model_hash

约定：
- hidden_index = encoder_layer + 1（hidden index 0 为 embedding，不导出为生产行）；
- 逐层头（B/C/D/E）导出 encoder layer 0-11 共 12 行/样本；
- 最终头 A（original_final_head）只导出 encoder layer 11 的 1 行/样本（完整模型基线）。

缓存（§5.3）：保存每层 pooled feature 而非整块 token hidden state，
NPZ + JSON 元数据，元数据含 model_hash / pooling，加载时版本不符即拒绝。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .heads import LayerHeadOutput, N_ENCODER_LAYERS

UNIFIED_COLUMNS = [
    "text_id", "hidden_index", "encoder_layer", "head_type", "pooling",
    "class_0_logit", "class_1_logit", "class_0_prob", "class_1_prob",
    "logit_margin_1_minus_0", "max_probability", "entropy", "model_hash",
]

# 生产逐层头（不包含 embedding 头；不包含只读最终头 A）
PRODUCTION_LAYER_HEADS = ["shared_frozen_head", "copied_layer_heads",
                          "random_layer_heads", "normalized_layer_heads"]
FINAL_HEAD = "original_final_head"


class CacheVersionError(Exception):
    """缓存版本与期望的 model_hash/pooling 不符。"""


# --------------------------------------------------------------------------- #
# 逐行展开
# --------------------------------------------------------------------------- #
def _binary_entropy(probs: torch.Tensor) -> torch.Tensor:
    """自然对数熵：-sum(p * log p)，[B, ...]。概率需有限且行和为 1。"""
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def layer_head_rows(
    output: LayerHeadOutput,
    text_ids: Sequence[str],
    model_hash: str,
    pooling: str,
    per_layer_heads: Optional[Iterable[str]] = None,
    include_final_head: bool = True,
) -> pd.DataFrame:
    """把逐层前向结果展开为统一 schema 的 DataFrame。

    per_layer_heads 缺省为 PRODUCTION_LAYER_HEADS 中当前模型已有的头；
    include_final_head=True 时把 original_final_head 作为完整模型基线导出 1 行/样本。
    """
    per_layer_heads = list(per_layer_heads) if per_layer_heads is not None else [
        h for h in PRODUCTION_LAYER_HEADS if h in output.results
    ]
    rows: List[Dict[str, Any]] = []
    n = len(text_ids)

    for head_type in per_layer_heads:
        logits = output.results[head_type]           # [B, 12, 2]
        probs = torch.softmax(logits, dim=-1)
        margins = logits[:, :, 1] - logits[:, :, 0]
        maxp = probs.max(dim=-1).values
        ent = _binary_entropy(probs)
        for i in range(n):
            for layer in range(N_ENCODER_LAYERS):
                rows.append({
                    "text_id": text_ids[i],
                    "hidden_index": layer + 1,
                    "encoder_layer": layer,
                    "head_type": head_type,
                    "pooling": pooling,
                    "class_0_logit": logits[i, layer, 0].item(),
                    "class_1_logit": logits[i, layer, 1].item(),
                    "class_0_prob": probs[i, layer, 0].item(),
                    "class_1_prob": probs[i, layer, 1].item(),
                    "logit_margin_1_minus_0": margins[i, layer].item(),
                    "max_probability": maxp[i, layer].item(),
                    "entropy": ent[i, layer].item(),
                    "model_hash": model_hash,
                })

    if include_final_head and FINAL_HEAD in output.results:
        logits = output.results[FINAL_HEAD]          # [B, 2]
        probs = torch.softmax(logits, dim=-1)
        margins = logits[:, 1] - logits[:, 0]
        maxp = probs.max(dim=-1).values
        ent = _binary_entropy(probs)
        layer = N_ENCODER_LAYERS - 1                  # encoder layer 11
        for i in range(n):
            rows.append({
                "text_id": text_ids[i],
                "hidden_index": layer + 1,
                "encoder_layer": layer,
                "head_type": FINAL_HEAD,
                "pooling": pooling,
                "class_0_logit": logits[i, 0].item(),
                "class_1_logit": logits[i, 1].item(),
                "class_0_prob": probs[i, 0].item(),
                "class_1_prob": probs[i, 1].item(),
                "logit_margin_1_minus_0": margins[i].item(),
                "max_probability": maxp[i].item(),
                "entropy": ent[i].item(),
                "model_hash": model_hash,
            })

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)


def expected_row_count(
    n_samples: int,
    per_layer_heads: int,
    include_final_head: bool,
) -> int:
    """按约定计算导出行数：n_samples * (12*per_layer_heads + int(include_final_head))。"""
    return n_samples * (N_ENCODER_LAYERS * per_layer_heads + int(include_final_head))


# --------------------------------------------------------------------------- #
# 特征缓存（每层 pooled feature）
# --------------------------------------------------------------------------- #
def _cache_meta_path(npz_path: str) -> str:
    return npz_path + ".meta.json"


def save_pooled_feature_cache(
    path: str,
    pooled_features: torch.Tensor,     # [N, 12, 768]
    text_ids: Sequence[str],
    attention_lengths: Sequence[int],
    pooling: str,
    model_hash: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """保存每层 pooled feature 到 NPZ + JSON 元数据。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pooled_np = pooled_features.detach().float().cpu().numpy()
    np.savez_compressed(
        path,
        pooled_features=pooled_np,
        text_ids=np.array(list(text_ids), dtype="U"),
        attention_lengths=np.array(attention_lengths, dtype=np.int64),
    )
    meta: Dict[str, Any] = {
        "pooling": pooling,
        "model_hash": model_hash,
        "shape": list(pooled_np.shape),
        "n_samples": len(text_ids),
        "schema_version": 1,
    }
    if extra:
        meta.update(extra)
    with open(_cache_meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def load_pooled_feature_cache(
    path: str,
    expected_pooling: Optional[str] = None,
    expected_model_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """加载缓存并校验版本；不符则抛 CacheVersionError。

    返回 {"pooled_features": np.ndarray [N,12,768], "text_ids": [...],
          "attention_lengths": [...], "meta": {...}}
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缓存不存在: {path}")
    meta_path = _cache_meta_path(path)
    if not os.path.isfile(meta_path):
        raise CacheVersionError(f"缓存缺少元数据: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if expected_pooling is not None and meta.get("pooling") != expected_pooling:
        raise CacheVersionError(
            f"pooling 不符: 缓存={meta.get('pooling')} 期望={expected_pooling}")
    if expected_model_hash is not None and meta.get("model_hash") != expected_model_hash:
        raise CacheVersionError(
            f"model_hash 不符: 缓存={meta.get('model_hash')} 期望={expected_model_hash}")

    data = np.load(path, allow_pickle=False)
    return {
        "pooled_features": data["pooled_features"],
        "text_ids": [str(t) for t in data["text_ids"]],
        "attention_lengths": data["attention_lengths"].tolist(),
        "meta": meta,
    }


# --------------------------------------------------------------------------- #
# 无标签诊断（§5.2，仅原始/共享冻结头）
# --------------------------------------------------------------------------- #
def layer_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """对原始/共享冻结头计算行为描述（非准确率，不能用于正式选层）。

    计算：相邻层 logits L2 变化、相邻层概率变化、各层与最终层预测一致率、
    margin/entropy 随层演化。输入为 layer_head_rows 的输出，且至少含
    shared_frozen_head 与 original_final_head。
    """
    rows: List[Dict[str, Any]] = []
    per_layer = df[df["head_type"] == "shared_frozen_head"].copy()
    final = df[df["head_type"] == "original_final_head"].copy()
    final_idx = final.set_index("text_id")
    per_layer["final_class"] = per_layer["text_id"].map(
        final_idx["class_0_logit"] < final_idx["class_1_logit"])
    per_layer["pred_class_1"] = per_layer["class_1_logit"] > per_layer["class_0_logit"]
    per_layer["consistent_with_final"] = (
        per_layer["pred_class_1"] == per_layer["final_class"]).astype(int)

    for text_id, grp in per_layer.groupby("text_id"):
        grp = grp.sort_values("encoder_layer")
        for idx in range(1, len(grp)):
            prev, cur = grp.iloc[idx - 1], grp.iloc[idx]
            rows.append({
                "text_id": text_id,
                "from_encoder_layer": int(prev["encoder_layer"]),
                "to_encoder_layer": int(cur["encoder_layer"]),
                "logit_l2_change": (
                    (cur["class_0_logit"] - prev["class_0_logit"]) ** 2
                    + (cur["class_1_logit"] - prev["class_1_logit"]) ** 2
                ) ** 0.5,
                "prob_change": abs(cur["class_0_prob"] - prev["class_0_prob"])
                + abs(cur["class_1_prob"] - prev["class_1_prob"]),
            })
    return pd.DataFrame(rows)
