#!/usr/bin/env python
"""P3 逐层输出导出：统一 schema CSV/Parquet + 每层 pooled feature 缓存。

默认使用固定 smoke-test 文本；也可用 --input 指定 JSONL/CSV（id,text）。

用法:
    conda activate 26intern
    python scripts/export_layer_outputs.py [--input texts.jsonl] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import load_yaml_config  # noqa: E402
from src.heads import build_layer_heads_model  # noqa: E402
from src.layer_outputs import (  # noqa: E402
    expected_row_count,
    layer_diagnostics,
    layer_head_rows,
    load_pooled_feature_cache,
    save_pooled_feature_cache,
)
from src.modeling import load_tokenizer, tokenize_texts  # noqa: E402

DEFAULT_TEXTS = [
    "北京天气怎么样，明天会下雨吗？",
    "这个项目的核心目标是提升模型的每一层利用率，而不是只用最后一层。",
    "今天股市大涨，投资者情绪明显回暖。",
    "报告指出风险加剧，建议谨慎观望。",
    "模型的能力不仅体现在最后一层的输出上，每一层都在逐步构建对输入的理解。",
]


def read_texts(path: str):
    """读取 JSONL/CSV，返回 (text_ids, texts)。"""
    if path.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        return [r["id"] for r in rows], [r["text"] for r in rows]
    df = pd.read_csv(path)
    return [str(i) for i in df["id"]], [str(t) for t in df["text"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="JSONL/CSV 文本文件（id,text）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    heads_cfg = load_yaml_config(os.path.join(ROOT, "configs", "heads.yaml"))
    ckpt = os.path.join(ROOT, cfg["paths"]["checkpoint"])
    base_dir = os.path.join(ROOT, cfg["paths"]["base_model_dir"])

    if args.input:
        text_ids, texts = read_texts(args.input)
    else:
        text_ids = [f"t{i}" for i in range(len(DEFAULT_TEXTS))]
        texts = DEFAULT_TEXTS

    enabled = heads_cfg["heads"]["enabled"]
    pooling = "masked_mean"
    model_hash = None  # 从 manifest 读取

    print(f"导出 {len(texts)} 条文本，heads={enabled}，pooling={pooling}")
    model = build_layer_heads_model(
        base_dir, ckpt, heads_enabled=enabled, pooling=pooling,
        pooling_confirmed=cfg["pooling_confirmed"], random_seed=42,
        model_hash=model_hash, device=args.device,
    ).eval()
    model_hash = model.model_hash

    tokenizer = load_tokenizer(base_dir)
    enc = tokenize_texts(texts, tokenizer, max_length=args.max_length)
    enc = {k: v.to(args.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)

    # ---- 统一 schema 导出 ----
    df = layer_head_rows(out, text_ids, model_hash, pooling)
    per_layer_heads = [h for h in ["shared_frozen_head", "copied_layer_heads",
                                   "random_layer_heads", "normalized_layer_heads"] if h in out.results]
    expected = expected_row_count(len(texts), len(per_layer_heads),
                                  include_final_head="original_final_head" in out.results)
    assert len(df) == expected, f"行数 {len(df)} != 期望 {expected}"
    assert (df["hidden_index"] == df["encoder_layer"] + 1).all()

    os.makedirs(os.path.join(ROOT, "reports", "tables"), exist_ok=True)
    csv_path = os.path.join(ROOT, "reports", "tables", "layer_outputs.csv")
    parquet_path = os.path.join(ROOT, "reports", "tables", "layer_outputs.parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"已写入 {csv_path}（{len(df)} 行）")
    print(f"已写入 {parquet_path}")

    # ---- 重读验证 ----
    df2 = pd.read_csv(csv_path)
    assert list(df2.columns) == list(df.columns)
    assert len(df2) == len(df)
    print(f"重读 CSV：行数 {len(df2)}，字段类型一致")

    # ---- 表征缓存 ----
    attn_lengths = enc["attention_mask"].sum(dim=1).tolist()
    cache_path = os.path.join(ROOT, "artifacts", "pooled_features.npz")
    save_pooled_feature_cache(cache_path, out.pooled_features, text_ids,
                              attn_lengths, pooling, model_hash)
    loaded = load_pooled_feature_cache(cache_path, expected_pooling=pooling,
                                       expected_model_hash=model_hash)
    print(f"缓存已写入并重读 {cache_path}: shape={loaded['pooled_features'].shape} "
          f"n_samples={loaded['meta']['n_samples']} pooling={loaded['meta']['pooling']}")

    # ---- 无标签诊断（共享/原始头）----
    diag = layer_diagnostics(df)
    if not diag.empty:
        diag_path = os.path.join(ROOT, "reports", "tables", "layer_diagnostics.csv")
        diag.to_csv(diag_path, index=False)
        print(f"诊断表已写入 {diag_path}（{len(diag)} 行）")

    print(f"\n结果: OK  pooling_confirmed={cfg['pooling_confirmed']}  model_hash={model_hash[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
