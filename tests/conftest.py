"""共享 fixtures：减少跨测试文件重复构建模型（BERT 加载较慢）。"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CKPT = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")
BASE_DIR = os.path.join(ROOT, "chinese-roberta-wwm-ext")

ALL_HEADS = ["original_final_head", "shared_frozen_head", "copied_layer_heads",
             "random_layer_heads", "normalized_layer_heads"]


@pytest.fixture(scope="session")
def base_dir():
    return BASE_DIR


@pytest.fixture(scope="session")
def ckpt_path():
    return CKPT


@pytest.fixture(scope="session")
def tokenizer(base_dir):
    from src.modeling import load_tokenizer
    return load_tokenizer(base_dir)


@pytest.fixture(scope="session")
def heads_model(base_dir, ckpt_path):
    """5 种 head 全开的逐层头模型（masked_mean pooling）。"""
    from src.heads import build_layer_heads_model
    return build_layer_heads_model(
        base_dir, ckpt_path, heads_enabled=ALL_HEADS, pooling="masked_mean",
        pooling_confirmed=False, random_seed=42, model_hash="test-hash", device="cpu",
    ).eval()
