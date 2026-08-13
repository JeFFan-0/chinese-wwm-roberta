# 模型（MODEL）

> 描述这个工程的模型结构、加载方式、逐层头与 Early-Exit 机制。代码在 `src/models/` 与 `src/probes/`。

## backbone

- 底座：`hfl/chinese-roberta-wwm-ext`，即 BERT 架构。
- 结构：12 个 encoder layer、hidden size 768、12 个 attention head、intermediate 3072、vocab 21128。
- 激活函数：GELU（不是 ReLU）。
- 微调 checkpoint `chinese-wwm-roberta.ckpt`：199 个 `bert.*` backbone 键 + `fc.weight [2,768]` + `fc.bias [2]`。

## 加载（可信、可审计）

- `src/models/checkpoint.py`：`load_state_dict_safe`（优先 `weights_only=True`）、`unwrap_state_dict`（解开 state_dict 包装）、`strip_prefix`（去 `module.`/`model.` 前缀）、`match_state_dicts`（matched/missing/unexpected/shape_mismatch + 参数覆盖率）。
- 加载结论：backbone 199/199 严格对齐，覆盖率 100%。

## 完整推理候选（`src/models/modeling.py`）

冻结 BERT backbone + 从 checkpoint 加载的原 `fc`，用 `pooling` 参数切换三种 pooling（见下）。
- 输出固定 `class_0_* / class_1_*`（标签映射未知，禁止命名正负）。

### pooling（`src/models/pooling.py`）

| 名称 | 公式 | 说明 |
|---|---|---|
| `cls` | `hidden[:, 0]` | 取 [CLS] 位 |
| `pooler` | `tanh(dense(CLS))` | 用 BERT 自带 pooler |
| `masked_mean` | 只对 `attention_mask==1` 的 token 求均值 | 抗 padding |

原模型实际用哪种**未确认**。

## 逐层模型头（`src/probes/heads.py`）

backbone 恒冻结，五种头：

| Head | 名称 | 说明 |
|---|---|---|
| A | `original_final_head` | 原始 `fc` 只读副本，仅作用最后层 |
| B | `shared_frozen_head` | 同一冻结 `fc` 依次作用 12 层 |
| C | `copied_layer_heads` | 12 个复制头（初始 = 原 fc），未来只训这些 |
| D | `random_layer_heads` | 12 个随机头（固定 seed 对照） |
| E | `normalized_layer_heads` | LayerNorm→Dropout→Linear（非默认） |

层编号：encoder layer 0–11；hidden index = encoder_layer + 1；embedding 头（hidden index 0）仅诊断。

## 逐层输出与缓存（`src/probes/layer_outputs.py`）

统一 schema 每行：
```
text_id, hidden_index, encoder_layer, head_type, pooling,
class_0_logit, class_1_logit, class_0_prob, class_1_prob,
logit_margin_1_minus_0, max_probability, entropy, model_hash
```
缓存存每层 pooled feature（NPZ + JSON 元数据，含 model_hash/pooling，版本不符即拒绝）。

## Early-Exit（`src/models/early_exit.py`）

真正逐层执行：每算一层就决定是否退出，**不会**先 `output_hidden_states=True` 跑完 12 层再挑层。
- 逐层：`bert.embeddings` → 构造 mask → 循环 `bert.encoder.layer[i]` → 候选层 pooling + head 得 logits → 算退出分数 → 满足即返回。
- layer 11 强制兜底；固定 layer 11 = 完整前向。
- 每个 encoder layer 挂 forward hook 计数，证明退出后不再调用后续层。

## 情绪因子三层协议（`src/factors/factor.py`）

1. `raw_predictions`：逐文本原始 logits/prob（不可变）；
2. `mapped_predictions`：加标签含义、主体、交易时间归属（标签映射确认后才含 p_negative/p_positive/sentiment_score）；
3. `daily_factor`：主体-交易日级聚合因子。

时间字段带时区，point-in-time 检查：盘后/周末文本归属下一交易日。
