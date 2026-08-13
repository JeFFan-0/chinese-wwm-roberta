"""P3 测试：特征缓存版本防护（test_cache_version_guard）。"""
import numpy as np
import pytest
import torch

from src.probes.layer_outputs import (
    CacheVersionError,
    load_pooled_feature_cache,
    save_pooled_feature_cache,
)


def test_cache_roundtrip(tmp_path):
    path = tmp_path / "pooled.npz"
    pooled = torch.randn(3, 12, 768)
    text_ids = ["a", "b", "c"]
    attn = [10, 20, 30]
    save_pooled_feature_cache(str(path), pooled, text_ids, attn,
                              pooling="masked_mean", model_hash="hash-abc")
    loaded = load_pooled_feature_cache(str(path), expected_pooling="masked_mean",
                                       expected_model_hash="hash-abc")
    assert loaded["pooled_features"].shape == (3, 12, 768)
    assert loaded["text_ids"] == ["a", "b", "c"]
    assert loaded["attention_lengths"] == [10, 20, 30]
    assert loaded["meta"]["pooling"] == "masked_mean"
    assert loaded["meta"]["model_hash"] == "hash-abc"
    assert np.allclose(loaded["pooled_features"], pooled.numpy(), atol=1e-6)


def test_cache_wrong_pooling_rejected(tmp_path):
    path = tmp_path / "pooled.npz"
    save_pooled_feature_cache(str(path), torch.randn(1, 12, 768), ["a"], [5],
                              pooling="cls", model_hash="h")
    with pytest.raises(CacheVersionError):
        load_pooled_feature_cache(str(path), expected_pooling="masked_mean")


def test_cache_wrong_hash_rejected(tmp_path):
    path = tmp_path / "pooled.npz"
    save_pooled_feature_cache(str(path), torch.randn(1, 12, 768), ["a"], [5],
                              pooling="cls", model_hash="hash-old")
    with pytest.raises(CacheVersionError):
        load_pooled_feature_cache(str(path), expected_model_hash="hash-new")


def test_cache_missing_meta_rejected(tmp_path):
    path = tmp_path / "pooled.npz"
    # 只写 npz，不写 meta
    np.savez_compressed(str(path), pooled_features=np.zeros((1, 12, 768)))
    with pytest.raises(CacheVersionError):
        load_pooled_feature_cache(str(path))


def test_cache_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pooled_feature_cache(str(tmp_path / "nope.npz"))


def test_cache_load_without_expected_is_allowed(tmp_path):
    path = tmp_path / "pooled.npz"
    save_pooled_feature_cache(str(path), torch.randn(1, 12, 768), ["a"], [5],
                              pooling="cls", model_hash="h")
    # 不指定期望版本也可以读取（调用方自行决定是否校验）
    loaded = load_pooled_feature_cache(str(path))
    assert loaded["meta"]["pooling"] == "cls"
