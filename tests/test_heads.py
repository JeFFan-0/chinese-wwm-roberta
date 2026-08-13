"""P2 测试：逐层模型头的初始化、冻结保证、shape 与复现性。

对应 TODO §11：test_head_initialization / test_random_head_seed /
test_backbone_frozen / test_layer_output_shape 等。
"""
import copy
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.checkpoint import load_state_dict_safe  # noqa: E402
from src.probes.heads import (  # noqa: E402
    CopiedLayerHeads,
    ExitCalibration,
    LayerHeadsModel,
    RandomLayerHeads,
    build_layer_heads_model,
)
from src.models.modeling import load_tokenizer, tokenize_texts  # noqa: E402
from src.models.pooling import apply_pooling  # noqa: E402

CKPT = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")
BASE_DIR = os.path.join(ROOT, "chinese-roberta-wwm-ext")

ALL_HEADS = ["original_final_head", "shared_frozen_head", "copied_layer_heads",
             "random_layer_heads", "normalized_layer_heads"]
TEXT = "北京天气怎么样，明天会下雨吗？"
OTHER = "这个项目的核心目标是提升模型的每一层利用率，而不是只用最后一层。"


@pytest.fixture(scope="module")
def ckpt_fc():
    state = load_state_dict_safe(CKPT)
    return state["fc.weight"], state["fc.bias"]


@pytest.fixture(scope="module")
def model():
    return build_layer_heads_model(
        BASE_DIR, CKPT, heads_enabled=ALL_HEADS, pooling="masked_mean",
        pooling_confirmed=False, random_seed=42, model_hash="test-hash", device="cpu",
    ).eval()


@pytest.fixture(scope="module")
def inputs(model):
    tokenizer = load_tokenizer(BASE_DIR)
    enc = tokenize_texts([TEXT, OTHER], tokenizer)
    return {k: v for k, v in enc.items()}


# --------------------------------------------------------------------------- #
# 初始化验证
# --------------------------------------------------------------------------- #
def test_original_final_head_matches_checkpoint(model, ckpt_fc):
    w, b = ckpt_fc
    head = model.original_final_head.fc
    assert torch.equal(head.weight.detach(), w)
    assert torch.equal(head.bias.detach(), b)


def test_copied_heads_init_match_fc(model, ckpt_fc):
    w, b = ckpt_fc
    for h in model.copied_layer_heads.heads:
        assert torch.equal(h.weight.detach(), w)
        assert torch.equal(h.bias.detach(), b)


def test_head_initialization_read_only_copies(model, inputs):
    """forward 后 original/shared 头参数不被修改（只读）。"""
    before_a = model.original_final_head.fc.weight.detach().clone()
    before_b = model.shared_frozen_head.head.weight.detach().clone()
    with torch.no_grad():
        model(**inputs)
    assert torch.equal(model.original_final_head.fc.weight.detach(), before_a)
    assert torch.equal(model.shared_frozen_head.head.weight.detach(), before_b)


def test_random_head_seed_reproducible():
    a = RandomLayerHeads(seed=123).heads
    b = RandomLayerHeads(seed=123).heads
    c = RandomLayerHeads(seed=456).heads
    for ha, hb, hc in zip(a, b, c):
        assert torch.equal(ha.weight, hb.weight)
        assert torch.equal(ha.bias, hb.bias)
        assert not torch.equal(ha.weight, hc.weight)


# --------------------------------------------------------------------------- #
# 共享头 vs 复制头初始一致性
# --------------------------------------------------------------------------- #
def test_shared_and_copied_identical_at_init(model, inputs):
    with torch.no_grad():
        out = model(**inputs)
    shared = out.results["shared_frozen_head"]     # [B, 12, 2]
    copied = out.results["copied_layer_heads"]     # [B, 12, 2]
    assert torch.allclose(shared, copied, atol=1e-6)


# --------------------------------------------------------------------------- #
# 输出 shape 与概率
# --------------------------------------------------------------------------- #
def test_layer_output_shape(model, inputs):
    with torch.no_grad():
        out = model(**inputs)
    for head_type in ["shared_frozen_head", "copied_layer_heads",
                      "random_layer_heads", "normalized_layer_heads"]:
        assert out.results[head_type].shape == (2, 12, 2), head_type
    assert out.results["original_final_head"].shape == (2, 2)


def test_probability_validity(model, inputs):
    with torch.no_grad():
        out = model(**inputs)
    for head_type in ["shared_frozen_head", "copied_layer_heads",
                      "random_layer_heads", "normalized_layer_heads"]:
        logits = out.results[head_type]          # [B,12,2]
        probs = torch.softmax(logits, dim=-1)
        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5)


def test_original_final_head_equals_final_layer_of_shared(model, inputs):
    """Head A 在最后层应与 shared 冻结头在最后层的输出一致（同一 fc）。"""
    with torch.no_grad():
        out = model(**inputs)
    a = out.results["original_final_head"]                 # [B,2]
    b = out.results["shared_frozen_head"][:, -1, :]        # [B,2]
    assert torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------- #
# 冻结保证
# --------------------------------------------------------------------------- #
def test_backbone_frozen(model):
    assert model.verify_backbone_frozen()
    assert all(not p.requires_grad for p in model.bert.parameters())


def test_synthetic_backward_backbone_grad_none(model, inputs):
    """合成反向传播：backbone 梯度全 None，可训练 head 有非零梯度。"""
    # 先确保无残留梯度
    model.zero_grad(set_to_none=True)
    loss = model(**inputs).results["copied_layer_heads"].sum()
    loss.backward()

    for name, p in model.bert.named_parameters():
        assert p.grad is None, f"backbone 参数 {name} 的 grad 应为 None"
    grads = [p.grad for p in model.copied_layer_heads.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_frozen_heads_have_no_grad(model, inputs):
    """original/shared 冻结头在混合反向传播中不产生梯度。"""
    model.zero_grad(set_to_none=True)
    out = model(**inputs)  # copied 头 requires_grad，允许反向
    # frozen 输出 + trainable 输出的混合标量：frozen 项为常数，仅 copied 有梯度
    loss = out.results["copied_layer_heads"].sum() + out.results["original_final_head"].sum()
    loss.backward()
    for p in model.original_final_head.parameters():
        assert p.grad is None
    for p in model.shared_frozen_head.parameters():
        assert p.grad is None


def test_trainable_head_parameters(model):
    params = model.trainable_head_parameters()
    assert len(params) > 0
    assert all(p.requires_grad for p in params)
    frozen_prefixes = ("original_final_head", "shared_frozen_head")
    trainable_ids = {id(p) for p in params}
    for n, p in model.named_parameters():
        if n.startswith(frozen_prefixes):
            assert id(p) not in trainable_ids, f"冻结头 {n} 不应出现在可训练列表中"
        if n.startswith("bert."):
            assert not p.requires_grad, f"backbone 参数 {n} 应冻结"


# --------------------------------------------------------------------------- #
# 参数量与开销记录
# --------------------------------------------------------------------------- #
def test_head_parameter_summary(model):
    summary = model.head_parameter_summary()
    for head_type in ALL_HEADS:
        assert head_type in summary
        assert summary[head_type]["param_count"] > 0
        assert summary[head_type]["storage_bytes"] > 0
    # copied = 12 * (768*2 + 2) 参数
    assert summary["copied_layer_heads"]["param_count"] == 12 * (768 * 2 + 2)


# --------------------------------------------------------------------------- #
# 层编号规范：embedding 不误标为 encoder layer 0
# --------------------------------------------------------------------------- #
def test_embedding_not_encoder_layer0(model, inputs):
    with torch.no_grad():
        out = model(**inputs)
    hs = out.hidden_states                      # 13 个
    assert len(hs) == 13
    pooled = out.pooled_features                # [B, 12, 768]
    # pooled[:, 0] 应对应 encoder layer 0 即 hidden index 1
    for i in range(12):
        expected = apply_pooling(model.pooling, hs[i + 1], attention_mask=inputs["attention_mask"],
                                 pooler=model.bert.pooler)
        assert torch.allclose(pooled[:, i], expected, atol=1e-5), f"layer {i}"

    # hidden index 0 (embedding) 不在 pooled_features 中
    emb = apply_pooling(model.pooling, hs[0], attention_mask=inputs["attention_mask"],
                        pooler=model.bert.pooler)
    assert not torch.allclose(emb, pooled[:, 0], atol=1e-2)


# --------------------------------------------------------------------------- #
# 校准占位符
# --------------------------------------------------------------------------- #
def test_calibration_placeholder():
    cal = ExitCalibration(n_layers=12)
    assert not cal.temperature_fit
    assert torch.all(cal.temperature == 1.0)
    logits = torch.randn(4, 2)
    scaled = cal.scale_logits(logits, 5)
    assert torch.allclose(scaled, logits, atol=1e-6)  # T=1 时不改变
    assert (cal.temperature > 0).all()


# --------------------------------------------------------------------------- #
# 选择性装配
# --------------------------------------------------------------------------- #
def test_heads_enabled_selection():
    m = build_layer_heads_model(BASE_DIR, CKPT, heads_enabled=["copied_layer_heads"],
                                pooling="cls", model_hash="h", device="cpu")
    assert hasattr(m, "copied_layer_heads")
    assert not hasattr(m, "original_final_head")
    assert not hasattr(m, "random_layer_heads")
    assert m.heads_enabled == ["copied_layer_heads"]


def test_deepcopy_heads_independent(model, ckpt_fc):
    """original/shared 头是独立副本，改其中一个不影响另一个。"""
    w, _ = ckpt_fc
    a = model.original_final_head.fc.weight.detach()
    b = model.shared_frozen_head.head.weight.detach()
    assert a.data_ptr() != b.data_ptr()
    assert torch.equal(a, w)
