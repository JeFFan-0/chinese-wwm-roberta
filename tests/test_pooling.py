"""P1 测试：三种 pooling、完整推理候选的确定性/一致性与概率有效性。

对应 TODO §11：test_pooling_shapes / test_masked_mean_padding /
test_probability_validity；以及 P1 验收中的单条 vs batch、重复运行一致、padding 不变。
CPU 运行；CUDA smoke 另设 marker。
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.checkpoint import load_state_dict_safe  # noqa: E402
from src.modeling import (  # noqa: E402
    BinaryClassificationCandidate,
    build_candidate,
    load_backbone,
    load_tokenizer,
    tokenize_texts,
)
from src.pooling import apply_pooling, masked_mean  # noqa: E402

CKPT = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")
BASE_DIR = os.path.join(ROOT, "chinese-roberta-wwm-ext")

TEXT = "北京天气怎么样，明天会下雨吗？"
OTHER = "这个项目的核心目标是提升模型的每一层利用率，而不是只用最后一层。"


@pytest.fixture(scope="module")
def assets():
    bert = load_backbone(BASE_DIR)
    state = load_state_dict_safe(CKPT)
    backbone = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    bert.load_state_dict(backbone, strict=True)
    tokenizer = load_tokenizer(BASE_DIR)
    return {"bert": bert, "state": state, "tokenizer": tokenizer}


@pytest.fixture(scope="module")
def make_candidate(assets):
    def _make(pooling: str) -> BinaryClassificationCandidate:
        fc = nn.Linear(768, 2)
        with torch.no_grad():
            fc.weight.copy_(assets["state"]["fc.weight"])
            fc.bias.copy_(assets["state"]["fc.bias"])
        return BinaryClassificationCandidate(
            assets["bert"], fc, pooling=pooling, pooling_confirmed=False, model_hash="test-hash",
        ).eval()

    return _make


@pytest.fixture
def text_batch(assets):
    enc = tokenize_texts([TEXT, OTHER], assets["tokenizer"])
    return {k: v for k, v in enc.items()}


# --------------------------------------------------------------------------- #
# 三种 pooling 的 shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pooling", ["cls", "pooler", "masked_mean"])
def test_pooling_shapes(pooling, make_candidate, text_batch):
    model = make_candidate(pooling)
    with torch.no_grad():
        out = model(**text_batch)
    assert out.logits.shape == (2, 2)
    assert out.probabilities.shape == (2, 2)
    assert out.pooled_feature.shape == (2, 768)


# --------------------------------------------------------------------------- #
# masked-mean 的 mask 语义
# --------------------------------------------------------------------------- #
def test_masked_mean_synthetic():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # [1, 3, 2]
    mask = torch.tensor([[1, 1, 0]])
    out = masked_mean(hidden, mask)
    expected = torch.tensor([[2.0, 3.0]])  # (1+3)/2, (2+4)/2
    assert torch.allclose(out, expected)


def test_masked_mean_padding_invariance(assets, make_candidate):
    """右侧增加 padding 不改变 masked-mean 结果。"""
    tok = assets["tokenizer"]
    single = tokenize_texts([TEXT], tok)                 # 无 padding（batch=1）
    batched = tokenize_texts([TEXT, OTHER], tok)         # 第一行被右侧 padding

    model = make_candidate("masked_mean")
    with torch.no_grad():
        p_single = model(**single).pooled_feature
        p_batch = model(**batched).pooled_feature
    assert torch.allclose(p_single[0], p_batch[0], atol=1e-5)


def test_cls_pooling_padding_invariance(assets, make_candidate):
    tok = assets["tokenizer"]
    single = tokenize_texts([TEXT], tok)
    batched = tokenize_texts([TEXT, OTHER], tok)
    model = make_candidate("cls")
    with torch.no_grad():
        p_single = model(**single).pooled_feature
        p_batch = model(**batched).pooled_feature
    assert torch.allclose(p_single[0], p_batch[0], atol=1e-6)


# --------------------------------------------------------------------------- #
# 概率有效性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pooling", ["cls", "pooler", "masked_mean"])
def test_probability_validity(pooling, make_candidate, text_batch):
    model = make_candidate(pooling)
    with torch.no_grad():
        out = model(**text_batch)
    probs = out.probabilities
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all() and (probs <= 1).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


# --------------------------------------------------------------------------- #
# 确定性 / 单条 vs batch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pooling", ["cls", "pooler", "masked_mean"])
def test_repeat_determinism(pooling, make_candidate, text_batch):
    model = make_candidate(pooling)
    with torch.no_grad():
        a = model(**text_batch).logits
        b = model(**text_batch).logits
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("pooling", ["cls", "pooler", "masked_mean"])
def test_single_vs_batch_consistency(pooling, assets, make_candidate):
    tok = assets["tokenizer"]
    model = make_candidate(pooling)
    with torch.no_grad():
        single = model(**tokenize_texts([TEXT], tok))
        batched = model(**tokenize_texts([TEXT, OTHER], tok))
    assert torch.allclose(single.class_0_logit, batched.class_0_logit[0], atol=1e-4)


# --------------------------------------------------------------------------- #
# 命名约束与错误处理
# --------------------------------------------------------------------------- #
def test_class_0_1_naming(make_candidate, text_batch):
    model = make_candidate("cls")
    with torch.no_grad():
        out = model(**text_batch)
    assert hasattr(out, "class_0_logit") and hasattr(out, "class_1_logit")
    assert hasattr(out, "class_0_prob") and hasattr(out, "class_1_prob")
    assert not hasattr(out, "p_positive") and not hasattr(out, "p_negative")


def test_unknown_pooling_raises(make_candidate, text_batch):
    model = make_candidate("cls")
    model.pooling = "bogus"
    with pytest.raises(ValueError):
        with torch.no_grad():
            model(**text_batch)


def test_pooler_pooling_requires_module():
    h = torch.randn(1, 8, 768)
    with pytest.raises(ValueError):
        apply_pooling("pooler", h)


def test_masked_mean_requires_mask():
    h = torch.randn(1, 8, 768)
    with pytest.raises(ValueError):
        apply_pooling("masked_mean", h)


# --------------------------------------------------------------------------- #
# 空白文本不产生 NaN
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pooling", ["cls", "pooler", "masked_mean"])
def test_blank_text_no_nan(pooling, assets, make_candidate):
    model = make_candidate(pooling)
    for blank in ["", "   ", "。"]:
        enc = tokenize_texts([blank], assets["tokenizer"])
        with torch.no_grad():
            out = model(**enc)
        assert torch.isfinite(out.logits).all(), f"pooling={pooling} blank={blank!r}"


# --------------------------------------------------------------------------- #
# 端到端 build_candidate（真实权重 + model_hash/pooling_confirmed 透传）
# --------------------------------------------------------------------------- #
def test_build_candidate_e2e():
    model = build_candidate(BASE_DIR, CKPT, pooling="cls", pooling_confirmed=False,
                            model_hash="dummy-hash", device="cpu")
    assert model.pooling == "cls"
    assert model.pooling_confirmed is False
    assert model.model_hash == "dummy-hash"
    # 输出字段固定命名
    tok = load_tokenizer(BASE_DIR)
    enc = tokenize_texts(["测试"], tok)
    with torch.no_grad():
        out = model(**enc)
    assert out.logits.shape == (1, 2)
    assert torch.isfinite(out.probabilities).all()
