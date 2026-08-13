#!/usr/bin/env python
"""P0 资产清点与可信加载：核验本地模型资产，生成 manifest 与参数匹配报告。

验收标准（TODO §2 与 §0.5）：
- 所有模型资产均有哈希、大小、来源和结构清单；
- backbone 严格加载且参数覆盖率 100%；
- fc.weight/fc.bias 状态被明确记录；
- 以退出码 0 在干净 shell 中完成。

用法:
    conda activate 26intern
    python scripts/verify_assets.py [--skip-hash]

路径来自 configs/model.yaml（支持 ${ENV:-default} 覆盖，不写死机器绝对路径）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.checkpoint import (  # noqa: E402
    load_state_dict_safe,
    match_state_dicts,
    report_fc,
    sha256_file,
    strip_prefix,
    unwrap_state_dict,
)
from src.config import load_yaml_config  # noqa: E402


# --------------------------------------------------------------------------- #
# 文件核验
# --------------------------------------------------------------------------- #
def verify_file(path: str, min_size: int = 1, label: str = "") -> Tuple[bool, Dict[str, Any]]:
    info: Dict[str, Any] = {"path": path, "exists": False, "readable": False,
                            "size_bytes": 0, "sha256": None, "label": label}
    ok = False
    if not os.path.exists(path):
        info["error"] = "file not found"
        return ok, info
    info["exists"] = True
    if not os.access(path, os.R_OK):
        info["error"] = "not readable"
        return ok, info
    info["readable"] = True
    size = os.path.getsize(path)
    info["size_bytes"] = size
    if size < min_size:
        info["error"] = f"size {size} < min_size {min_size}"
        return ok, info
    ok = True
    return ok, info


# --------------------------------------------------------------------------- #
# 环境信息
# --------------------------------------------------------------------------- #
def env_info() -> Dict[str, Any]:
    import transformers

    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["device_name"] = torch.cuda.get_device_name(0)
    else:
        info["cuda"] = None
        info["cudnn"] = None
        info["device_name"] = None
    return info


# --------------------------------------------------------------------------- #
# 模型结构核验
# --------------------------------------------------------------------------- #
def build_backbone(base_dir: str):
    from transformers import BertModel

    model = BertModel.from_pretrained(base_dir, local_files_only=True)
    return model


def backbone_config_report(model) -> Dict[str, Any]:
    cfg = model.config
    expected = {
        "model_type": cfg.model_type,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "hidden_act": cfg.hidden_act,
        "pad_token_id": cfg.pad_token_id,
    }
    return expected


# --------------------------------------------------------------------------- #
# Tokenizer 核验
# --------------------------------------------------------------------------- #
def tokenizer_report(base_dir: str, cfg) -> Tuple[bool, Dict[str, Any]]:
    from transformers import BertTokenizerFast

    tok = BertTokenizerFast.from_pretrained(base_dir, local_files_only=True)
    report: Dict[str, Any] = {
        "vocab_size": tok.vocab_size,
        "pad_token": tok.pad_token,
        "pad_token_id": tok.pad_token_id,
        "cls_token": tok.cls_token,
        "cls_token_id": tok.cls_token_id,
        "sep_token": tok.sep_token,
        "sep_token_id": tok.sep_token_id,
        "unk_token": tok.unk_token,
        "unk_token_id": tok.unk_token_id,
        "mask_token": tok.mask_token,
        "mask_token_id": tok.mask_token_id,
    }
    ok = True
    checks = []
    if tok.vocab_size != cfg.vocab_size:
        ok = False
        checks.append(f"vocab_size mismatch: tokenizer={tok.vocab_size} config={cfg.vocab_size}")
    if tok.pad_token_id != cfg.pad_token_id:
        ok = False
        checks.append(f"pad_token_id mismatch: tokenizer={tok.pad_token_id} config={cfg.pad_token_id}")
    if tok.pad_token_id is None:
        ok = False
        checks.append("pad_token_id is None")
    report["checks"] = checks
    return ok, report


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="P0 资产清点与可信加载")
    parser.add_argument("--skip-hash", action="store_true", help="跳过大文件 SHA-256（测试加速）")
    args = parser.parse_args()

    cfg = load_yaml_config(os.path.join(ROOT, "configs", "model.yaml"))
    paths = cfg["paths"]
    ckpt_path = os.path.join(ROOT, paths["checkpoint"])
    base_dir = os.path.join(ROOT, paths["base_model_dir"])

    errors: List[str] = []
    manifest: Dict[str, Any] = {}

    t0 = time.time()
    print("=" * 70)
    print("P0 资产清点与可信加载")
    print("=" * 70)

    # ---- 1. 文件核验 ----
    print("\n[1] 文件核验")
    files_to_check = [
        (ckpt_path, 100_000_000, "checkpoint"),
        (os.path.join(base_dir, "pytorch_model.bin"), 100_000_000, "base_weights"),
        (os.path.join(base_dir, "config.json"), 100, "base_config"),
        (os.path.join(base_dir, "vocab.txt"), 10_000, "vocab"),
        (os.path.join(base_dir, "tokenizer.json"), 1_000, "tokenizer"),
    ]
    file_info: Dict[str, Dict[str, Any]] = {}
    for path, min_size, label in files_to_check:
        ok, info = verify_file(path, min_size=min_size, label=label)
        file_info[label] = info
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {label:<16} {path}  ({info['size_bytes']} bytes)")
        if not ok:
            errors.append(f"{label}: {info.get('error')}")
        elif not args.skip_hash:
            print(f"         hashing ...")
            info["sha256"] = sha256_file(path)
            print(f"         sha256={info['sha256'][:16]}...")

    # ---- 2. 环境 ----
    print("\n[2] 环境版本")
    env = env_info()
    for k, v in env.items():
        print(f"  {k:<18} {v}")

    # ---- 3. 结构核验 ----
    print("\n[3] 模型结构核验")
    base_model = None
    backbone_cfg = {}
    try:
        base_model = build_backbone(base_dir)
        backbone_cfg = backbone_config_report(base_model)
        for k, v in backbone_cfg.items():
            print(f"  config.{k:<22} {v}")
        # 与 configs/model.yaml 期望对比
        expected = cfg["base_model"]
        for k in ("num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size", "vocab_size"):
            if backbone_cfg.get(k) != expected.get(k):
                errors.append(f"config field {k}: local={backbone_cfg.get(k)} expected={expected.get(k)}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"backbone build failed: {exc}")
        print(f"  [FAIL] {exc}")

    # ---- 4. Tokenizer 核验 ----
    print("\n[4] Tokenizer 核验")
    tok_ok = True
    tok_report = {}
    if base_model is not None:
        try:
            tok_ok, tok_report = tokenizer_report(base_dir, base_model.config)
            for k, v in tok_report.items():
                if k != "checks":
                    print(f"  tokenizer.{k:<16} {v}")
            if tok_report.get("checks"):
                print("  FAIL:", tok_report["checks"])
                errors.extend(tok_report["checks"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tokenizer build failed: {exc}")
            print(f"  [FAIL] {exc}")

    # ---- 5. 参数匹配报告 ----
    print("\n[5] 参数匹配报告")
    key_report: Dict[str, Any] = {}
    try:
        ckpt_raw = load_state_dict_safe(ckpt_path, map_location="cpu")
        ckpt_state = unwrap_state_dict(ckpt_raw)
        if not isinstance(ckpt_state, dict):
            raise TypeError(f"unwrap 后不是 dict: {type(ckpt_state)}")
        ckpt_state = strip_prefix(ckpt_state)

        if base_model is not None:
            model_state = base_model.state_dict()

            # 5a. ckpt backbone 严格加载到 BertModel
            # ckpt 键带 "bert." 前缀（训练代码保存的是整体模型），BertModel.state_dict()
            # 键不带前缀，因此这里去掉 "bert." 前缀后再匹配 —— 即 check.ipynb 已证明的结论。
            ckpt_backbone = {k[len("bert."):]: v for k, v in ckpt_state.items() if k.startswith("bert.")}
            report_strict = match_state_dicts(ckpt_backbone, model_state)
            print(f"  backbone strict load: matched={report_strict['matched_key_count']} "
                  f"missing={report_strict['missing_key_count']} "
                  f"unexpected={report_strict['unexpected_key_count']} "
                  f"shape_mismatch={report_strict['shape_mismatch_count']}")
            print(f"    tensor coverage={report_strict['matched_tensor_ratio']:.6f}  "
                  f"param coverage={report_strict['matched_param_ratio']:.6f}")
            if report_strict["missing_keys"]:
                print("    missing:", report_strict["missing_keys"][:10])
                errors.append(f"backbone missing keys: {report_strict['missing_keys'][:10]}")
            if report_strict["unexpected_keys"]:
                print("    unexpected:", report_strict["unexpected_keys"][:10])
                # strict 加载会因 unexpected 键失败，P0 验收必须显式报错
                errors.append(f"backbone unexpected keys: {report_strict['unexpected_keys'][:10]}")
            if report_strict["shape_mismatch"]:
                print("    shape mismatch:", report_strict["shape_mismatch"][:5])
                errors.append("backbone shape mismatch > 0")
            if report_strict["matched_param_ratio"] < 1.0:
                errors.append("backbone param coverage < 100%")

            # 5b. ckpt vs 底座全量（含 cls.* 头与 fc）
            base_state = load_state_dict_safe(os.path.join(base_dir, "pytorch_model.bin"), map_location="cpu")
            base_state = unwrap_state_dict(base_state)
            base_state = strip_prefix(base_state)
            report_full = match_state_dicts(ckpt_state, base_state)
            print(f"  ckpt vs base full: matched={report_full['matched_key_count']} "
                  f"missing={report_full['missing_key_count']} "
                  f"unexpected={report_full['unexpected_key_count']} "
                  f"shape_mismatch={report_full['shape_mismatch_count']}")
            print("    only in base (MLM/NSP heads):", [k for k in report_full["missing_keys"][:8]])
            print("    only in ckpt:", report_full["unexpected_keys"])

            # 5c. fc 状态
            fc = report_fc(ckpt_state)
            print(f"  fc.weight: present={fc['fc.weight']['present']} shape={fc['fc.weight']['shape']}")
            print(f"  fc.bias:   present={fc['fc.bias']['present']} shape={fc['fc.bias']['shape']}")

            key_report = {
                "backbone_strict_load": report_strict,
                "ckpt_vs_base_full": report_full,
                "fc": fc,
            }
    except Exception as exc:  # noqa: BLE001
        errors.append(f"key report failed: {exc}")
        print(f"  [FAIL] {exc}")

    # ---- 6. 落盘 manifest 与键报告 ----
    manifest = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": {
            "path": paths["checkpoint"],
            "sha256": file_info.get("checkpoint", {}).get("sha256"),
            "size_bytes": file_info.get("checkpoint", {}).get("size_bytes", 0),
            "source": "unknown",
        },
        "base_model": {
            "path": paths["base_model_dir"],
            "name": cfg["base_model"]["name"],
            "revision": cfg["base_model"]["revision"],
            "weights_sha256": file_info.get("base_weights", {}).get("sha256"),
            "config_sha256": file_info.get("base_config", {}).get("sha256"),
            "tokenizer_sha256": file_info.get("tokenizer", {}).get("sha256"),
        },
        "environment": env,
        "backbone_config": backbone_cfg,
        "tokenizer": tok_report,
        "key_report": {
            "backbone_strict_load": {
                "matched_key_count": key_report.get("backbone_strict_load", {}).get("matched_key_count"),
                "missing_key_count": key_report.get("backbone_strict_load", {}).get("missing_key_count"),
                "unexpected_key_count": key_report.get("backbone_strict_load", {}).get("unexpected_key_count"),
                "shape_mismatch_count": key_report.get("backbone_strict_load", {}).get("shape_mismatch_count"),
                "matched_tensor_ratio": key_report.get("backbone_strict_load", {}).get("matched_tensor_ratio"),
                "matched_param_ratio": key_report.get("backbone_strict_load", {}).get("matched_param_ratio"),
            },
            "fc": key_report.get("fc", {}),
        },
    }

    os.makedirs(os.path.join(ROOT, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "reports", "tables"), exist_ok=True)
    with open(os.path.join(ROOT, "metadata", "model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print("\n[6] 写入 metadata/model_manifest.json")

    # ---- 7. 键报告 CSV/JSON 落盘 ----
    if key_report:
        with open(os.path.join(ROOT, "reports", "tables", "checkpoint_key_report.json"), "w", encoding="utf-8") as f:
            json.dump(key_report, f, ensure_ascii=False, indent=2, default=str)
        _write_key_report_csv(key_report)
        print("    写入 reports/tables/checkpoint_key_report.json / .csv")

    # ---- 结果 ----
    print("\n" + "=" * 70)
    if errors:
        print(f"结果: FAIL  ({len(errors)} 个错误)")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"结果: OK   用时 {time.time() - t0:.1f}s")
    print("=" * 70)
    return 0


def _write_key_report_csv(key_report: Dict[str, Any]) -> None:
    import csv

    out = os.path.join(ROOT, "reports", "tables", "checkpoint_key_report.csv")
    rows: List[List[str]] = []
    strict = key_report.get("backbone_strict_load", {})
    rows.append(["report", "backbone_strict_load", "", "", ""])
    for k in ("matched_key_count", "missing_key_count", "unexpected_key_count",
              "shape_mismatch_count", "matched_tensor_ratio", "matched_param_ratio"):
        rows.append(["metric", k, str(strict.get(k)), "", ""])
    rows.append(["", "", "", "", ""])
    for status, keys in (("matched", strict.get("matched_keys", [])),
                         ("missing", strict.get("missing_keys", [])),
                         ("unexpected", strict.get("unexpected_keys", []))):
        for key in keys:
            rows.append([status, key, "", "", ""])
    for m in strict.get("shape_mismatch", []):
        rows.append(["shape_mismatch", m.get("key", ""), str(m.get("reference_shape")),
                     str(m.get("candidate_shape")), ""])
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "key", "metric_or_ref_shape", "candidate_shape", "note"])
        writer.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
