"""P0 测试：checkpoint 解包、前缀处理、键匹配与严格加载（test_checkpoint_identity /
test_key_unwrap / test_strict_backbone_load 等）。"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.checkpoint import (  # noqa: E402
    load_state_dict_safe,
    match_state_dicts,
    param_count,
    report_fc,
    sha256_file,
    sha256_tensor,
    state_dict_diff_metrics,
    strip_prefix,
    unwrap_state_dict,
)

CKPT = os.path.join(ROOT, "chinese-wwm-roberta.ckpt")
BASE_DIR = os.path.join(ROOT, "chinese-roberta-wwm-ext")


# --------------------------------------------------------------------------- #
# 解包与前缀处理
# --------------------------------------------------------------------------- #
def test_key_unwrap_single_level():
    inner = {"a": 1, "b": 2}
    wrapped = {"state_dict": inner}
    assert unwrap_state_dict(wrapped) is inner


def test_key_unwrap_nested():
    inner = {"a": 1}
    wrapped = {"model_state_dict": {"state_dict": inner}}
    assert unwrap_state_dict(wrapped) is inner


def test_key_unwrap_untouched_passthrough():
    plain = {"a": torch.zeros(1)}
    assert unwrap_state_dict(plain) is plain


def test_strip_prefix_removes_and_keeps_order():
    sd = {"module.bert.embeddings.weight": torch.ones(1), "model.fc.weight": torch.ones(2),
          "no_prefix.key": torch.ones(3)}
    out = strip_prefix(sd, prefixes=("module.", "model."))
    assert set(out.keys()) == {"bert.embeddings.weight", "fc.weight", "no_prefix.key"}


def test_strip_prefix_does_not_mutate_input():
    sd = {"module.a": torch.ones(1)}
    before = dict(sd)
    strip_prefix(sd)
    assert set(sd.keys()) == set(before.keys())
    assert sd["module.a"].data_ptr() == before["module.a"].data_ptr()


def test_key_unwrap_real_ckpt():
    state = load_state_dict_safe(CKPT)
    assert "fc.weight" in state and "fc.bias" in state


# --------------------------------------------------------------------------- #
# 自比较恒等（test_checkpoint_identity）
# --------------------------------------------------------------------------- #
def test_checkpoint_identity_self_compare():
    state = load_state_dict_safe(CKPT)
    metrics = state_dict_diff_metrics(state, state)
    assert metrics["delta_l2"] == 0.0
    assert metrics["max_abs_delta"] == 0.0
    assert metrics["mean_cosine"] == pytest.approx(1.0)
    assert metrics["mae"] == 0.0


# --------------------------------------------------------------------------- #
# 严格加载（test_strict_backbone_load）
# --------------------------------------------------------------------------- #
def test_strict_backbone_load_real_model():
    from transformers import BertModel

    model = BertModel.from_pretrained(BASE_DIR, local_files_only=True)
    state = load_state_dict_safe(CKPT)
    backbone = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    report = match_state_dicts(backbone, model.state_dict())
    assert report["missing_key_count"] == 0
    assert report["unexpected_key_count"] == 0
    assert report["shape_mismatch_count"] == 0
    assert report["matched_tensor_ratio"] == pytest.approx(1.0)
    assert report["matched_param_ratio"] == pytest.approx(1.0)
    # 严格加载本身成功
    model.load_state_dict(backbone, strict=True)


# --------------------------------------------------------------------------- #
# 键匹配报告
# --------------------------------------------------------------------------- #
def test_match_state_dicts_basic():
    ref = {"a": torch.zeros(2), "b": torch.zeros(3)}
    cand = {"a": torch.zeros(2), "c": torch.ones(4)}
    report = match_state_dicts(cand, ref)
    assert report["matched_key_count"] == 1
    assert report["matched_keys"] == ["a"]
    assert report["missing_keys"] == ["b"]
    assert report["unexpected_keys"] == ["c"]
    assert report["shape_mismatch_count"] == 0
    assert report["matched_tensor_ratio"] == pytest.approx(0.5)


def test_match_state_dicts_shape_mismatch():
    ref = {"a": torch.zeros(2)}
    cand = {"a": torch.zeros(3)}
    report = match_state_dicts(cand, ref)
    assert report["shape_mismatch_count"] == 1
    assert report["matched_key_count"] == 0
    assert report["shape_mismatch"][0]["key"] == "a"


def test_match_param_ratio_weights_by_numel():
    ref = {"small": torch.zeros(1), "large": torch.zeros(100)}
    cand = {"small": torch.zeros(1), "large": torch.ones(100)}
    report = match_state_dicts(cand, ref)
    # 全部匹配，numel 覆盖率 100%
    assert report["matched_param_ratio"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# fc 报告与哈希
# --------------------------------------------------------------------------- #
def test_report_fc_real_ckpt():
    state = load_state_dict_safe(CKPT)
    fc = report_fc(state)
    assert fc["fc.weight"]["present"] is True
    assert fc["fc.weight"]["shape"] == [2, 768]
    assert fc["fc.bias"]["present"] is True
    assert fc["fc.bias"]["shape"] == [2]


def test_report_fc_absent():
    state = {"x": torch.zeros(1)}
    fc = report_fc(state)
    assert fc["fc.weight"]["present"] is False


def test_sha256_file_deterministic(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert sha256_file(str(p)) == sha256_file(str(p))


def test_sha256_tensor_deterministic():
    t = torch.randn(16, 768)
    assert sha256_tensor(t) == sha256_tensor(t.clone())
    assert sha256_tensor(t) != sha256_tensor(t + 1e-9)


def test_param_count():
    assert param_count({"a": torch.zeros(10), "b": torch.zeros(5)}) == 15


def test_load_state_dict_safe_missing_file():
    with pytest.raises(FileNotFoundError):
        load_state_dict_safe("/nonexistent/path.ckpt")
