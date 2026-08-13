#!/usr/bin/env python
"""将本地权重转换为 safetensors 格式（TODO §2.4 安全要求）。

原因：transformers 5.x 出于 CVE-2025-32434 限制，torch<2.6 时拒绝通过
``from_pretrained`` 读取 .bin 权重；safetensors 不受该限制，且更安全、更快。

用法:
    conda activate 26intern
    python scripts/convert_to_safetensors.py            # 转换底座权重
    python scripts/convert_to_safetensors.py --ckpt     # 同时转换微调 ckpt

转换是**新增文件**，不删除原始 .bin/.ckpt。输出：
    chinese-roberta-wwm-ext/model.safetensors           （底座）
    artifacts/chinese-wwm-roberta.safetensors           （微调 ckpt，artifacts/ 已 gitignore）
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.checkpoint import unwrap_state_dict  # noqa: E402

BASE_BIN = os.path.join(ROOT, "chinese-roberta-wwm-ext", "pytorch_model.bin")
BASE_OUT = os.path.join(ROOT, "chinese-roberta-wwm-ext", "model.safetensors")
CKPT_IN = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")
CKPT_OUT = os.path.join(ROOT, "artifacts", "chinese-wwm-roberta.safetensors")


def convert(src: str, dst: str, label: str) -> None:
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{label} 不存在: {src}")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    print(f"转换 {label}: {src} -> {dst}")
    state = torch.load(src, map_location="cpu", weights_only=True)
    state = unwrap_state_dict(state)
    # 解码器与 word_embeddings 共享内存（tied embeddings），safetensors 禁止；
    # 统一 clone 成独立张量，保证重载后逐元素一致。
    cloned = {k: v.detach().clone().contiguous() for k, v in state.items()}
    save_file(cloned, dst)
    size = os.path.getsize(dst)
    print(f"  完成: {size / 1e6:.1f} MB, {len(state)} 个张量")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="store_true", help="同时转换微调 ckpt")
    args = parser.parse_args()

    convert(BASE_BIN, BASE_OUT, "底座权重")
    if args.ckpt:
        convert(CKPT_IN, CKPT_OUT, "微调 ckpt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
