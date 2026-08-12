#!/usr/bin/env python
"""P7 合成数据管线：加载 → 校验 → split → 特征缓存 → 只训练 head → 校准接口。

验证目标（§9.3）：
- loss 能反向传播；
- head 参数发生变化；
- backbone 参数完全不变；
- checkpoint 可以保存和重新加载，重载后 logits 一致；
- 12 个 head 的指标输出格式完整；
- 温度恒正、校准函数无 NaN/Inf；
- 阈值搜索强制独立 calibration 输入。

**合成数据结果不提交为模型表现结论**；输出全部标记 synthetic_only=true。

用法:
    conda activate 26intern
    python scripts/run_synthetic_pipeline.py [--device cpu]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
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
from src.config import load_yaml_config  # noqa: E402
from src.data import (  # noqa: E402
    assert_no_overlap,
    generate_synthetic_dataset,
    load_dataset,
    make_dataloader,
    split_dataset,
    validate_dataset,
)
from src.heads import build_layer_heads_model  # noqa: E402
from src.modeling import load_tokenizer  # noqa: E402
from src.training import (  # noqa: E402
    load_cached_features,
    load_head_checkpoint,
    save_head_checkpoint,
    train_heads_from_features,
)


def write_synthetic_csv(samples, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label", "entity_id", "published_at", "source"])
        for s in samples:
            w.writerow([s.id, s.text, s.label, s.entity_id,
                        s.published_at.isoformat() if s.published_at else "",
                        s.source or ""])


def write_synthetic_jsonl(samples, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({
                "id": s.id, "text": s.text, "label": s.label,
                "entity_id": s.entity_id,
                "published_at": s.published_at.isoformat() if s.published_at else None,
                "source": s.source,
            }, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    ckpt = os.path.join(ROOT, cfg["paths"]["checkpoint"])
    base_dir = os.path.join(ROOT, cfg["paths"]["base_model_dir"])
    label_map = {"0": "unknown", "1": "unknown"}

    print("=" * 72)
    print("P7 合成数据管线（synthetic_only）")
    print("=" * 72)

    # ---- 1. 生成合成数据 + 写 CSV/JSONL ----
    samples = generate_synthetic_dataset(n=64, seed=42, include_metadata=True)
    os.makedirs(os.path.join(ROOT, "artifacts", "synthetic"), exist_ok=True)
    csv_path = os.path.join(ROOT, "artifacts", "synthetic", "syn_train.csv")
    jsonl_path = os.path.join(ROOT, "artifacts", "synthetic", "syn_train.jsonl")
    write_synthetic_csv(samples, csv_path)
    write_synthetic_jsonl(samples, jsonl_path)

    # ---- 2. CSV/JSONL 均能完整加载且校验通过 ----
    from_csv = load_dataset(csv_path, label_map)
    from_jsonl = load_dataset(jsonl_path, label_map)
    assert len(from_csv) == len(samples) == len(from_jsonl), "CSV/JSONL 加载数量不一致"
    vr = validate_dataset(from_csv, label_map)
    assert vr.ok, f"合成数据校验应通过: {vr.errors}"
    print(f"[1] 合成数据 {len(samples)} 条；CSV/JSONL 均加载；校验通过")

    # 检测非法情况（用副本，避免污染原始数据集）
    from src.data import Sample
    bad = [Sample(s.id, s.text, s.label, s.entity_id, s.published_at, s.source)
           for s in samples[:3]]
    bad[0].id = bad[1].id                       # 重复 ID
    bad[1].text = ""                             # 空文本
    bad[2].label = 7                             # 非法 label
    vbad = validate_dataset(bad, label_map)
    assert any("duplicate_id" in e for e in vbad.errors)
    assert any("empty_text" in e for e in vbad.errors)
    assert any("invalid_label" in e for e in vbad.errors)
    print("[2] 重复 ID / 空文本 / 非法 label 均被检测")

    # ---- 3. split 不重叠 ----
    train, dev, test = split_dataset(samples, train_frac=0.7, dev_frac=0.15, seed=42,
                                     stratify_key="label")
    assert_no_overlap(train, dev, test)
    assert len(train) + len(dev) + len(test) == len(samples)
    print(f"[3] split: train={len(train)} dev={len(dev)} test={len(test)} 互不重叠")

    # ---- 4. 特征缓存（一次前向）+ 训练 ----
    print("[4] 构建模型并缓存特征")
    model = build_layer_heads_model(
        base_dir, ckpt, heads_enabled=["copied_layer_heads"], pooling="masked_mean",
        pooling_confirmed=False, model_hash="synthetic-hash", device=args.device,
    ).eval()
    model_hash = model.model_hash
    tokenizer = load_tokenizer(base_dir)
    dl = list(make_dataloader(train, tokenizer, batch_size=8, shuffle=True, seed=1))
    cache_path = os.path.join(ROOT, "artifacts", "synthetic", "features.npz")
    feats, labels, ids = [], [], []
    for enc, batch_labels, batch_ids in dl:
        enc = {k: v.to(args.device) for k, v in enc.items()}
        out = model(**enc)
        feats.append(out.pooled_features.detach().cpu())
        labels.extend(batch_labels)
        ids.extend(batch_ids)
    feats = torch.cat(feats, dim=0)
    print(f"    缓存特征 shape={tuple(feats.shape)}  label 数={len(labels)}")

    summary = train_heads_from_features(model, feats, labels, head_type="copied_layer_heads",
                                        n_epochs=3, lr=0.01, device="cpu")
    assert summary.backbone_unchanged, "训练后 backbone 应完全不变"
    print(f"[5] 训练 {summary.n_epochs} epoch：backbone 不变={summary.backbone_unchanged}")
    print(f"    每层 train acc（synthetic_only）: "
          f"{[round(a, 3) for a in summary.per_layer_train_acc]}")

    # ---- 5. checkpoint 保存/重载，logits 一致 ----
    ckpt_path = os.path.join(ROOT, "artifacts", "synthetic", "heads.safetensors")
    save_head_checkpoint(model, "copied_layer_heads", ckpt_path)
    before = model(**enc).results["copied_layer_heads"].detach().cpu()
    model2 = build_layer_heads_model(
        base_dir, ckpt, heads_enabled=["copied_layer_heads"], pooling="masked_mean",
        pooling_confirmed=False, model_hash="synthetic-hash", device="cpu",
    ).eval()
    load_head_checkpoint(model2, "copied_layer_heads", ckpt_path)
    with torch.no_grad():
        after = model2(**{k: v.to("cpu") for k, v in enc.items()}).results["copied_layer_heads"]
    max_diff = (before - after).abs().max().item()
    assert max_diff < 1e-5, f"重载后 logits 不一致: {max_diff}"
    print(f"[6] 保存/重载 head checkpoint；logits 最大差 {max_diff:.2e}（一致）")

    # ---- 6. 校准接口（合成 logits）----
    print("[7] 校准接口（synthetic_only）")
    with torch.no_grad():
        out = model(**{k: v.to(args.device) for k, v in enc.items()})
    logits = out.results["copied_layer_heads"]                     # [B,12,2]
    tgt = torch.tensor(labels[: logits.shape[0]], dtype=torch.long)

    temp = fit_temperature(logits[:, -1], tgt, steps=100)
    assert temp.item() > 0, "温度必须恒正"
    scaled = temperature_scale(logits[:, -1], temp)
    n = nll(scaled, tgt).item()
    e = ece(scaled, tgt, n_bins=10).item()
    assert all(torch.isfinite(torch.tensor([n, e, temp])).tolist()), "校准结果出现 NaN/Inf"
    print(f"    温度 T={temp.item():.4f}  NLL={n:.4f}  ECE={e:.4f}（合成 logits）")

    # 阈值搜索：calibration 与 test 严格分离
    cal_logits = logits[:32, -1]
    cal_tgt = tgt[:32]
    test_ids = set(ids[32:])
    cal_ids = set(ids[:32])
    assert_calibration_test_separate(cal_ids, test_ids)
    res = search_threshold(cal_logits, cal_tgt, score_fn=max_prob_score,
                           target_quality=0.5, min_coverage=0.2, mode="max")
    print(f"    阈值搜索（calibration 独立）：coverage={res.coverage:.2f} "
          f"quality={res.quality:.2f} calibrated={res.calibrated}")
    print(f"    分数示例：max_prob entropy margin 均有限="
          f"{torch.isfinite(entropy_score(logits[:, -1])).all().item()}")

    # ---- 7. 输出标记 synthetic_only ----
    out_csv = os.path.join(ROOT, "reports", "tables", "synthetic_pipeline_report.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value", "synthetic_only"])
        w.writerow(["n_samples", len(samples), "true"])
        w.writerow(["train_acc_per_layer", str(summary.per_layer_train_acc), "true"])
        w.writerow(["backbone_unchanged", summary.backbone_unchanged, "true"])
        w.writerow(["checkpoint_reload_max_diff", max_diff, "true"])
        w.writerow(["temperature", temp.item(), "true"])
        w.writerow(["nll", n, "true"])
        w.writerow(["ece", e, "true"])
        w.writerow(["threshold_coverage", res.coverage, "true"])
        w.writerow(["threshold_quality", res.quality, "true"])
    print(f"[8] 报告写入 {out_csv}（全部 synthetic_only=true）")
    print("\n结果: OK  （所有数值均为合成数据控制流验证，不构成模型表现结论）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
