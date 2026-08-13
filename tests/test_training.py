"""P7 测试：只训练 heads 的合成管线（test_synthetic_train_heads_only）。"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.dataset import generate_synthetic_dataset, make_dataloader  # noqa: E402
from src.models.modeling import load_tokenizer, tokenize_texts  # noqa: E402
from src.probes.training import (  # noqa: E402
    load_cached_features,
    load_head_checkpoint,
    save_head_checkpoint,
    train_heads_from_features,
)

TEXT_A = "北京天气怎么样，明天会下雨吗？"


@pytest.fixture(scope="module")
def synthetic_features(heads_model, tokenizer):
    """从真实模型一次前向得到合成缓存的 pooled features。"""
    samples = generate_synthetic_dataset(n=24, seed=5, include_metadata=False)
    texts = [s.text for s in samples]
    enc = tokenize_texts(texts, tokenizer)
    with torch.no_grad():
        out = heads_model(**enc)
    labels = [s.label for s in samples]
    return out.pooled_features, labels, [s.id for s in samples]


# --------------------------------------------------------------------------- #
# test_synthetic_train_heads_only
# --------------------------------------------------------------------------- #
def test_synthetic_train_heads_only(heads_model, synthetic_features):
    feats, labels, _ = synthetic_features
    # 记录 head 与 backbone 权重
    head_before = heads_model.copied_layer_heads.heads[0].weight.detach().clone()
    backbone_before = [p.detach().clone() for p in heads_model.bert.parameters()]

    summary = train_heads_from_features(heads_model, feats, labels,
                                        head_type="copied_layer_heads",
                                        n_epochs=2, lr=0.01, batch_size=8, device="cpu")
    assert summary.synthetic_only
    assert summary.backbone_unchanged
    # head 参数确实变化
    head_after = heads_model.copied_layer_heads.heads[0].weight.detach()
    assert not torch.equal(head_before, head_after)
    # backbone 完全不变
    for before, after in zip(backbone_before, heads_model.bert.parameters()):
        assert torch.equal(before, after.detach())
    # 12 层指标输出完整
    assert len(summary.per_layer_train_loss) == 12
    assert len(summary.per_layer_train_acc) == 12
    assert all(0.0 <= a <= 1.0 for a in summary.per_layer_train_acc)


def test_optimizer_only_heads(heads_model, synthetic_features):
    """训练接口只接收 head 参数。"""
    feats, labels, _ = synthetic_features
    module = heads_model.copied_layer_heads
    params = [p for p in module.parameters() if p.requires_grad]
    assert all(p.requires_grad for p in params)
    # backbone 全冻结
    assert all(not p.requires_grad for p in heads_model.bert.parameters())


# --------------------------------------------------------------------------- #
# checkpoint 保存/重载，logits 一致
# --------------------------------------------------------------------------- #
def test_checkpoint_save_reload_logits(heads_model, synthetic_features, tmp_path):
    feats, labels, ids = synthetic_features
    train_heads_from_features(heads_model, feats, labels, head_type="copied_layer_heads",
                              n_epochs=1, lr=0.01, device="cpu")
    enc = tokenize_texts([TEXT_A], load_tokenizer(os.path.join(ROOT, "chinese-roberta-wwm-ext")))
    with torch.no_grad():
        before = heads_model(**enc).results["copied_layer_heads"].detach().cpu()

    ckpt_path = str(tmp_path / "heads.safetensors")
    save_head_checkpoint(heads_model, "copied_layer_heads", ckpt_path)

    from src.probes.heads import build_layer_heads_model
    model2 = build_layer_heads_model(
        os.path.join(ROOT, "chinese-roberta-wwm-ext"),
        os.path.join(ROOT, "chinese-wwm-roberta.ckpt"),
        heads_enabled=["copied_layer_heads"], pooling="masked_mean", model_hash="h",
        device="cpu",
    ).eval()
    load_head_checkpoint(model2, "copied_layer_heads", ckpt_path)
    with torch.no_grad():
        after = model2(**enc).results["copied_layer_heads"]
    assert torch.allclose(before, after, atol=1e-5)


# --------------------------------------------------------------------------- #
# 缓存版本保护（训练头前必须核对 model hash 与 pooling）
# --------------------------------------------------------------------------- #
def test_training_cache_version_guard(tmp_path, heads_model, synthetic_features, tokenizer):
    feats, labels, ids = synthetic_features
    attn_lens = [len(t) for t in ids]
    from src.probes.layer_outputs import save_pooled_feature_cache
    from src.probes.layer_outputs import CacheVersionError

    path = str(tmp_path / "f.npz")
    save_pooled_feature_cache(path, feats, ids, attn_lens, pooling="masked_mean",
                              model_hash="hash-A", extra={"labels": labels})
    ok = load_cached_features(path, expected_pooling="masked_mean", expected_model_hash="hash-A")
    assert ok[0].shape == feats.shape
    with pytest.raises(CacheVersionError):
        load_cached_features(path, expected_pooling="cls", expected_model_hash="hash-A")
    with pytest.raises(CacheVersionError):
        load_cached_features(path, expected_pooling="masked_mean", expected_model_hash="hash-B")
