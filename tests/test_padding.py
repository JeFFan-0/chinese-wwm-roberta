"""padding 不变性测试：右侧 padding / batch 组成不影响相同文本结果。

对应 TODO §11 test_masked_mean_padding 的模型级版本，以及 P3 验收
"batch 顺序变化不会改变相同 text_id 的结果"。
"""
import torch

from src.modeling import tokenize_texts

TEXT_A = "北京天气怎么样，明天会下雨吗？"
TEXT_B = "这是一个很长的测试句子，用于制造右侧 padding。" * 8  # 强制 TEXT_A 被 padding
TEXT_C = "短"


def _pooled_row(model, tokenizer, texts, index):
    enc = tokenize_texts(texts, tokenizer)
    with torch.no_grad():
        out = model(**enc)
    return out.pooled_features[index]


def test_right_padding_does_not_change_masked_mean(heads_model, tokenizer):
    # 单独跑 TEXT_A（batch=1，无 padding）
    solo = _pooled_row(heads_model, tokenizer, [TEXT_A], 0)
    # 与长文本组批，TEXT_A 被右侧 padding
    batched = _pooled_row(heads_model, tokenizer, [TEXT_A, TEXT_B], 0)
    assert torch.allclose(solo, batched, atol=1e-4)


def test_extra_padding_invariance(heads_model, tokenizer):
    # 在更长 batch 中，TEXT_C 被 padding 更多
    p2 = _pooled_row(heads_model, tokenizer, [TEXT_C, TEXT_A], 0)
    p3 = _pooled_row(heads_model, tokenizer, [TEXT_C, TEXT_A, TEXT_B], 0)
    assert torch.allclose(p2, p3, atol=1e-4)


def test_batch_composition_invariance_full_output(heads_model, tokenizer):
    """相同文本在不同 batch 组成下，逐层 logits 一致。"""
    from src.layer_outputs import layer_head_rows

    enc1 = tokenize_texts([TEXT_A, TEXT_B], tokenizer)
    enc2 = tokenize_texts([TEXT_B, TEXT_A], tokenizer)
    with torch.no_grad():
        out1 = heads_model(**enc1)
        out2 = heads_model(**enc2)
    for head_type, t1 in out1.results.items():
        t2 = out2.results[head_type]
        # t1 的第 0 行（TEXT_A）应等于 t2 的第 1 行（TEXT_A）
        idx1, idx2 = (0, 1) if head_type != "original_final_head" else (0, 1)
        assert torch.allclose(t1[idx1], t2[idx2], atol=1e-4), head_type
