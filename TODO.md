# 中文 RoBERTa 无数据阶段执行方案与验收清单

> 项目：`chinese-wwm-roberta`
>
> 适用阶段：已经取得微调 checkpoint 和本地预训练底座，但暂时没有原任务训练、验证或测试数据。
>
> 总目标：在不修改 BERT 各层权重、不依赖标签数据的前提下，完成可信加载、完整模型推理、逐层模型头、真实 Early-Exit、性能基准、未来数据接口与情绪因子输出协议。数据到位后只需训练模型头、校准阈值和评估，不再重构基础工程。

---

## 0. 当前边界与完成定义

### 0.1 当前已知事实

- backbone 是 BERT 架构：12 个 encoder layer、hidden size 768、12 个 attention head、intermediate size 3072、vocab size 21128。
- checkpoint 中存在 `bert.*` backbone 参数。
- checkpoint 中存在输出维度为 2 的 `fc.weight`，说明它包含一个二分类头候选。
- `check.ipynb` 已证明去掉 `bert.` 前缀后，backbone 参数可加载到本地模型结构。
- 本地底座配置的激活函数是 GELU，而不是 ReLU。

### 0.2 当前仍未知

- `class 0` 和 `class 1` 分别代表什么，不能提前命名为负面/正面。
- 原模型头接收的是 CLS、`pooler_output`、masked mean pooling，还是其他特征。
- 原前向在分类头之前是否包含 dropout、LayerNorm、激活函数或额外层。
- 原 tokenizer 参数、最大长度、截断方向和数据清洗规则。
- 当前本地底座是否就是微调时实际使用的底座 revision。
- 模型的真实 Accuracy、Macro-F1、校准性和因子有效性。

### 0.3 无数据阶段允许交付的成果

- 可复现、可审计的 checkpoint 和底座加载流程；
- 一个可配置 pooling 的完整二分类推理候选模型；
- 原始最终头、共享冻结头、逐层复制头、逐层随机对照头；
- `[batch, 12, 2]` 的逐层 logits/概率输出；
- 真正停止后续 encoder layer 计算的 Early-Exit 引擎；
- 固定层退出的真实 latency、吞吐和显存基准；
- 数据、训练、校准和情绪因子输出接口；
- 自动测试和验收报告。

### 0.4 无数据阶段禁止声称的结论

- 不得声称某一层分类效果最好或已包含足够情绪信息。
- 不得选择正式 Early-Exit 阈值。
- 不得报告 Accuracy、Macro-F1 或因子预测能力。
- 不得把低置信度解释为“中性”。二分类模型没有中性类别。
- 不得在标签映射未知时把 `class 0/1` 命名为 negative/positive。
- 不得把最终层头直接作用于浅层的失败解释为“浅层没有信息”。
- 不得把固定层的理论计算减少直接等同于真实部署加速。

### 0.5 本阶段 Done 定义

只有同时满足以下条件，才视为无数据阶段完成：

- [ ] 所有模型资产均有哈希、大小、来源和结构清单。
- [ ] backbone 严格加载且参数覆盖率 100%。
- [ ] `fc` 参数清单和所有候选 pooling 路径已实现。
- [ ] 12 层模型头能够一次输出统一格式结果。
- [ ] 强制最后层退出与普通完整前向在数值容差内一致。
- [ ] 强制浅层退出时，后续层通过执行计数证明未被调用。
- [ ] 固定层性能矩阵已在目标硬件上测量并落盘。
- [ ] 数据接入、只训练头、校准和因子输出接口通过合成数据测试。
- [ ] README/报告明确记录未知项、限制和数据到位后的下一步。

---

## 1. 推荐工程结构

现有 `check.ipynb` 保留为探索记录；核心逻辑从 notebook 中移出：

```text
chinese-wwm-roberta/
├── README.md
├── plan.md
├── TODO.md
├── requirements.txt                  # 或 environment.yml / uv.lock
├── configs/
│   ├── model.yaml                    # 路径、pooling、label 占位符
│   ├── heads.yaml                    # head 类型和初始化策略
│   ├── early_exit.yaml               # 候选层、退出规则占位符
│   └── benchmark.yaml                # 长度、batch、重复次数、硬件配置
├── metadata/
│   ├── model_manifest.json
│   └── data_schema.json
├── src/
│   ├── checkpoint.py                 # 安全解包、前缀处理、键报告
│   ├── modeling.py                   # backbone 与完整模型候选
│   ├── pooling.py                    # CLS/pooler/masked mean
│   ├── heads.py                      # 各类逐层模型头
│   ├── layer_outputs.py              # 逐层输出与导出格式
│   ├── early_exit.py                 # 真正逐层执行
│   ├── factor.py                     # 因子输出协议与聚合接口
│   └── data.py                       # CSV/JSONL/合成数据
├── scripts/
│   ├── verify_assets.py
│   ├── smoke_inference.py
│   ├── export_layer_outputs.py
│   ├── benchmark_fixed_exit.py
│   └── run_synthetic_pipeline.py
├── tests/
│   ├── test_checkpoint.py
│   ├── test_pooling.py
│   ├── test_layer_mapping.py
│   ├── test_heads.py
│   ├── test_early_exit.py
│   ├── test_padding.py
│   └── test_factor_schema.py
├── reports/
│   ├── tables/
│   ├── figures/
│   └── no_data_stage_report.md
└── artifacts/                        # 权重、缓存和大产物，不提交 GitHub
```

### 验收

- [ ] 所有核心代码均能被普通 Python 脚本导入，不依赖 notebook 隐藏状态。
- [ ] checkpoint、底座权重、缓存特征和训练产物均在 `.gitignore` 中。
- [ ] 配置中使用相对路径或环境变量，不写死某台机器的绝对路径。
- [ ] notebook 只调用 `src/` 并绘图，不复制核心实现。

---

## 2. P0：资产清点与可信加载

**优先级：最高。依赖：无。**

### 2.1 实现资产核验脚本

创建 `scripts/verify_assets.py`，检查：

- checkpoint 是否存在、可读、文件大小是否合理；
- 底座 config、tokenizer 和权重是否齐全；
- checkpoint、底座权重、config、tokenizer 的 SHA-256；
- Python、PyTorch、Transformers、CUDA、cuDNN 版本；
- config 中的 `model_type`、层数、hidden size、head 数、FFN size、vocab size；
- tokenizer 的 vocab size、pad/CLS/SEP/UNK token ID 是否与 config 一致。

生成 `metadata/model_manifest.json`，推荐字段：

```json
{
  "checkpoint": {
    "path": "chinese-wwm-roberta.ckpt",
    "sha256": "...",
    "size_bytes": 0,
    "source": "unknown"
  },
  "base_model": {
    "path": "chinese-roberta-wwm-ext",
    "revision": "unknown",
    "weights_sha256": "...",
    "config_sha256": "...",
    "tokenizer_sha256": "..."
  },
  "environment": {
    "python": "...",
    "torch": "...",
    "transformers": "...",
    "cuda": "..."
  }
}
```

### 2.2 统一 checkpoint 解包

在 `src/checkpoint.py` 中实现纯函数：

```python
def unwrap_state_dict(obj):
    for key in ("state_dict", "model_state_dict"):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
    return obj

def strip_prefix(state_dict, prefixes=("module.", "model.")):
    ...
```

注意：当前 `compare_layers.py` 中在 `for d in (base, ft)` 内给 `d` 重新赋值，不会修改 `base` 或 `ft`，必须改成显式返回和赋值。

### 2.3 输出参数匹配报告

对 backbone 和分类头分别报告：

- matched keys；
- missing keys；
- unexpected keys；
- shape mismatches；
- matched tensor ratio；
- matched parameter-count ratio；
- `fc.weight`、`fc.bias` 是否存在及 shape。

将完整结果写入 `reports/tables/checkpoint_key_report.csv/json`。

### 2.4 安全要求

- 只加载可信来源的 checkpoint。
- PyTorch 版本允许时优先 `torch.load(..., weights_only=True)`。
- 若后续重新保存，优先转换为 safetensors。
- 禁止把 checkpoint 二进制提交 GitHub。

### P0 验收

- [ ] `verify_assets.py` 在干净 shell 中以退出码 0 完成。
- [ ] `model_manifest.json` 含所有必需哈希和环境版本。
- [ ] backbone 以 `strict=True` 成功加载。
- [ ] backbone tensor coverage = 100%，parameter coverage = 100%。
- [ ] shape mismatch = 0。
- [ ] `fc.weight`/`fc.bias` 状态被明确记录，而非默认假设。
- [ ] 用相同 state dict 进行自比较时，差异为 0。

### P0 交付物

- `src/checkpoint.py`
- `scripts/verify_assets.py`
- `metadata/model_manifest.json`
- `reports/tables/checkpoint_key_report.*`
- 对应单元测试

---

## 3. P1：重建完整二分类推理候选

**优先级：最高。依赖：P0。**

由于原模型类和训练代码暂缺，本阶段实现“可枚举、可替换、明确标注为候选”的前向，而不是武断认定某种 pooling。

### 3.1 实现三种 pooling

在 `src/pooling.py` 中实现统一接口：

1. `cls`：`last_hidden_state[:, 0]`；
2. `pooler`：BERT 的 `pooler_output`；
3. `masked_mean`：只对 `attention_mask == 1` 的 token 求均值。

masked mean 参考：

```python
def masked_mean(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom
```

### 3.2 实现完整模型候选

`src/modeling.py` 至少返回：

- `logits`；
- `probabilities`；
- pooled feature；
- pooling 名称；
- model/checkpoint hash；
- 可选的 hidden states。

在标签映射未知时，字段固定为：

```text
class_0_logit
class_1_logit
class_0_prob
class_1_prob
logit_margin_1_minus_0
```

禁止提前命名为 positive/negative。

### 3.3 Smoke test 文本

使用固定小集合覆盖：

- 短文本；
- 长文本；
- 空白/近空文本的明确处理；
- 中英文数字和标点；
- 两条 `check.ipynb` 中现有示例；
- 组批后产生 padding 的不同长度文本。

测试只验证工程稳定性，不评价情绪语义是否正确。

### 3.4 原模型恢复接口

在配置中保留：

```yaml
pooling: unknown
label_mapping:
  "0": unknown
  "1": unknown
pre_classifier:
  dropout: unknown
  layer_norm: unknown
```

原训练代码到位后，可直接填写而不用重写模型。

### P1 验收

- [ ] 三种 pooling 都能对 batch 正确输出 `[batch, 768]`。
- [ ] 分类头输出 `[batch, 2]`，概率逐行和约等于 1。
- [ ] 相同输入、相同配置重复运行 logits 在容差内一致。
- [ ] 单条推理与放入 batch 后的对应结果在容差内一致。
- [ ] 增加右侧 padding 不改变 CLS/masked-mean 结果。
- [ ] 未知标签始终以 `class_0/1` 输出。
- [ ] 配置和运行日志明确标记当前 pooling 是否已经确认。

### P1 交付物

- `src/pooling.py`
- `src/modeling.py`
- `scripts/smoke_inference.py`
- 三种 pooling 的 smoke-test 输出
- 对应单元测试

---

## 4. P2：准备逐层模型头

**优先级：最高。依赖：P1。backbone 始终冻结。**

### 4.1 层编号规范

模型返回 13 个 hidden states：

| hidden index | 含义 | encoder layer |
|---:|---|---:|
| 0 | embedding 输出 | N/A |
| 1 | 第一个 encoder 输出 | 0 |
| ... | ... | ... |
| 12 | 最后一个 encoder 输出 | 11 |

生产候选只对 12 个 encoder layer 接头。embedding 头仅作诊断，默认不进入 Early-Exit。

### 4.2 Head A：原始最终头

名称：`original_final_head`

- 从 checkpoint 加载原 `fc` 参数；
- 永远保留一份只读副本；
- 不训练；
- 用于完整模型基线和最后层一致性测试。

### 4.3 Head B：共享冻结头

名称：`shared_frozen_head`

- 同一个原始 `fc` 依次作用于 12 层 pooled feature；
- backbone 和 head 都不更新；
- 用于零训练兼容性测试。

解释边界：它衡量浅层特征与最终层分类平面的兼容程度，不衡量浅层情绪信息的上限。

### 4.4 Head C：12 个复制线性头

名称：`copied_layer_heads`

- 为 encoder layer 0–11 各创建独立 `Linear(768, 2)`；
- 初始参数均从原始 `fc` 复制；
- 当前只创建、序列化和执行前向；
- 数据到位后只优化这些 heads，不修改 backbone；
- 最后一层头可保持冻结原始版本，另外创建可训练副本，避免覆盖基线。

### 4.5 Head D：12 个随机线性头

名称：`random_layer_heads`

- 相同结构，使用固定 seed 随机初始化；
- 为未来复制初始化实验提供对照；
- 保存初始化 seed 和配置；
- 未训练前不得解释输出语义。

### 4.6 Head E：轻量增强候选

名称：`normalized_layer_heads`

```text
LayerNorm(768) → Dropout(p) → Linear(768, 2)
```

- 当前只实现，不作为默认头；
- 用于未来验证不同层尺度差异是否影响线性头；
- 配置化 dropout，默认 0.1；
- 必须与纯 Linear 做同数据、同预算对照。

### 4.7 校准参数占位符

- 为每个出口准备正值温度参数 `T_l`，初始值为 1；
- 当前不拟合温度；
- 当前不保存任何“推荐阈值”；
- 配置中的阈值必须标记为 `smoke_test_only`。

### 4.8 冻结保证

实现并测试：

```python
backbone.requires_grad_(False)
backbone.eval()
```

训练接口只允许：

```python
optimizer = AdamW(model.heads.parameters(), ...)
```

测试一次合成反向传播：所有 backbone 参数的 `grad is None`，head 至少有一个非零梯度。

### P2 验收

- [ ] `original_final_head` 参数逐元素等于 checkpoint `fc`。
- [ ] 共享头和复制头初始状态在每层输出完全一致。
- [ ] 随机头可由 seed 精确复现。
- [ ] 所有生产层头输出 shape 为 `[batch, 12, 2]`。
- [ ] backbone 所有参数 `requires_grad=False`。
- [ ] 合成反向传播后 backbone 梯度全为 `None`。
- [ ] head 参数量和额外存储开销被记录。
- [ ] embedding 头不会被误标成 encoder layer 0。

### P2 交付物

- `src/heads.py`
- `configs/heads.yaml`
- heads 初始化清单和参数量报告
- 对应单元测试

---

## 5. P3：逐层输出、导出与无标签诊断

**优先级：高。依赖：P2。**

### 5.1 统一前向输出

对每条文本和每个 encoder layer 输出：

```text
text_id
hidden_index
encoder_layer
head_type
pooling
class_0_logit
class_1_logit
class_0_prob
class_1_prob
logit_margin_1_minus_0
max_probability
entropy
model_hash
```

### 5.2 无标签可做的诊断

仅针对原始/共享冻结头，可计算：

- 相邻层 logits L2 变化；
- 相邻层概率变化；
- 各层预测类别与最终层的一致率；
- margin/entropy 随层演化；
- 输出对 padding 和 batch 组成的不变性。

这些是行为描述，不是准确率，也不能用于正式选择层。

### 5.3 表征缓存接口

实现缓存：

- 保存每层 pooled feature，而不是整块 token hidden state，降低存储；
- 保存 `text_id`、attention length、pooling、model hash；
- 缓存格式优先 Parquet/NPZ/safetensors；
- 缓存文件写入 `artifacts/` 并忽略 Git；
- 未来训练头时必须核对缓存的 model hash 和 pooling。

### P3 验收

- [ ] 输出行数 = 样本数 × 12 × head 类型数。
- [ ] `hidden_index = encoder_layer + 1`。
- [ ] 概率有效，无 NaN/Inf，逐行和约为 1。
- [ ] 导出后重读数据，字段类型与行数不变。
- [ ] batch 顺序变化不会改变相同 `text_id` 的结果。
- [ ] 缓存包含 model hash 和 pooling，错误版本缓存会被拒绝。

### P3 交付物

- `src/layer_outputs.py`
- `scripts/export_layer_outputs.py`
- 输出 schema
- 小型 smoke-test CSV/Parquet
- 对应单元测试

---

## 6. P4：真正逐层执行的 Early-Exit 引擎

**优先级：最高。依赖：P2。**

### 6.1 关键原则

以下写法不能获得加速：

```python
out = bert(..., output_hidden_states=True)
# BERT 已经完成全部 12 层，再选择某个头
```

真正 Early-Exit 必须在每层执行之后决定是否继续。

### 6.2 逐层执行流程

在 `src/early_exit.py` 中：

1. 执行 embeddings；
2. 使用模型原生 helper 构造 extended attention mask；
3. 循环调用 `bert.encoder.layer[i]`；
4. 在配置的候选层对 hidden state 做相同 pooling；
5. 调用对应 head 得到 logits；
6. 计算退出分数；
7. 满足条件立即返回，不再执行后续层；
8. 最后层强制退出。

优先复用当前 Transformers 版本的模型内部 mask/encoder 约定，避免手工假设 mask dtype 和广播语义。

### 6.3 当前实现的退出策略

#### 固定层退出

```text
exit when encoder_layer == k
```

这是无数据阶段唯一可用于正式性能基准的策略。

#### 最大概率阈值

```text
max(softmax(logits)) >= threshold
```

只用于控制流 smoke test。未校准 softmax 不能用于正式部署。

#### Margin 阈值

二分类下可使用概率差或 logit 差；同样只用于 smoke test。

#### 最终层兜底

无论阈值如何，encoder layer 11 必须输出结果。

### 6.4 执行追踪

返回：

```text
logits
probabilities
exit_layer
executed_layer_count
exit_reason
latency_ms
```

为测试增加 layer call counter 或 forward hook，记录每一层被执行次数。

### 6.5 Batch 策略

第一版先支持：

- 整个 batch 在同一固定层退出；
- batch size 1 的动态退出。

第二版再实现 active-set batching：已退出样本从 batch 移除，剩余样本继续下一层。必须保持样本索引映射，最终恢复原顺序。

### P4 验收

- [ ] 固定 layer 11 退出时，logits 与普通完整模型在 `atol/rtol` 容差内一致。
- [ ] 固定 layer k 退出时，k+1 至 11 的 call counter 均为 0。
- [ ] `executed_layer_count == exit_layer + 1`。
- [ ] 极高阈值时所有样本由最后层兜底。
- [ ] 极低阈值 smoke test 时样本能在首个候选层退出。
- [ ] active-set batching 输出顺序与输入顺序一致。
- [ ] 不同 batch 大小下无 NaN/Inf。
- [ ] 所有测试均在 CPU 运行；有 CUDA 时再增加 GPU smoke test。

### P4 交付物

- `src/early_exit.py`
- `configs/early_exit.yaml`
- 固定层和动态控制流示例
- 执行追踪日志
- 对应单元测试

---

## 7. P5：无数据性能基准

**优先级：中高。依赖：P4。**

### 7.1 测试矩阵

至少覆盖：

- sequence length：32、64、128、256；
- batch size：1、8、16、32；
- 固定退出层：2、4、6、8、10、11；
- device：目标 GPU；CPU 若为部署候选也需测试；
- dtype：FP32；如果生产会使用 FP16/BF16，再单独测量。

### 7.2 基准方法

- 每个配置先预热至少 10 次；
- 正式运行至少 50–100 次；
- GPU 测量前后调用 `torch.cuda.synchronize()`；
- 禁用梯度并使用 eval 模式；
- 固定随机输入或固定 tokenizer 输出；
- 同时记录普通完整模型作为基线；
- 不把首次模型加载和 tokenizer 时间混入纯模型前向延迟；
- 另行记录端到端文本到输出延迟。

### 7.3 输出指标

- p50/p95 latency；
- samples/s；
- peak allocated GPU memory；
- executed layer count；
- 相对完整 12 层的速度比；
- 理论层数节省比例；
- 理论节省与实测加速之间的差距。

### 7.4 结果命名

基准只能表述为：

> 如果未来证明可以在 layer k 退出，则在硬件 H、batch B、长度 L 下，测得速度为 X。

不能表述为：

> layer k 已足够完成情绪任务。

### P5 验收

- [ ] 测试矩阵全部完成或明确记录跳过原因。
- [ ] 每行结果包含硬件、软件版本、batch、长度、dtype、退出层。
- [ ] 完整模型基线在同一进程、相同配置下测量。
- [ ] 报告 p50/p95，而不是单次耗时。
- [ ] GPU 时间使用同步方法测量。
- [ ] 结果可由同一配置重复运行，主要指标波动在预设容差内。

### P5 交付物

- `scripts/benchmark_fixed_exit.py`
- `configs/benchmark.yaml`
- `reports/tables/fixed_exit_benchmark.csv`
- latency/throughput 图
- 基准环境说明

---

## 8. P6：修正现有分析的可信度问题

**优先级：高。依赖：P0。**

### 8.1 权重统计

对每张量增加：

- `numel`；
- base L2；
- delta L2；
- relative L2；
- mean absolute delta；
- **真实 max absolute delta**；
- cosine similarity。

层级聚合必须按参数本身计算：

```text
delta_l2_layer = sqrt(Σ ||ΔW_p||²)
base_l2_layer  = sqrt(Σ ||W_base,p||²)
rel_l2_layer   = delta_l2_layer / base_l2_layer
mae_layer      = Σ ||ΔW_p||₁ / Σ numel(p)
max_layer      = max |ΔW_p|
```

不得再对每个张量的 `rel_l2` 简单平均，因为小 bias 和大型权重矩阵不应等权。

### 8.2 层编号

- hidden index 0 是 embedding；
- encoder 层只编号 0–11；
- 图和表同时保存 `hidden_index` 与 `encoder_layer`；
- 删除任何“encoder layer 12”的表述。

### 8.3 Masking

- token 激活统计只对 `attention_mask == 1` 求值；
- attention 比较同时 mask 无效 query 和 key；
- 同时提供按 token 的 micro 平均和先按句平均的 macro 平均；
- 测试增加额外 padding 后统计不变。

### 8.4 结论措辞

- hidden-state 差异随深度扩大，应称为“累计表示差异”；
- 不由累计曲线推断某一层自身改动最大；
- 不用 Q/K/V 范数直接判断 head 重要性；
- 不使用“ReLU 死神经元”，因为配置是 GELU；
- 未计算真实 max delta 前，不报告“最大绝对误差”；
- 没有标签数据时，不报告层任务价值或情绪能力。

### 8.5 无数据可搭建的局部比较接口

可以实现但仅在自选文本上做 smoke test：

- 同一 hidden 输入分别通过 base 和 finetuned 的对应层；
- 输出局部 cosine/L2；
- base/finetuned 层替换的完整前向接口。

在没有代表性数据时不形成总体结论。

### P6 验收

- [ ] 相同 state dict 自比较时所有 delta 为 0、cos 为 1。
- [ ] 层级结果可从张量级原始值精确复算。
- [ ] padding invariance 测试通过。
- [ ] 所有图只使用 encoder layer 0–11。
- [ ] notebook 小结删除或降级所有无证据结论。
- [ ] 从干净 kernel 顺序执行通过。

### P6 交付物

- 修订后的 `compare_layers.py`
- 修订后的 `check.ipynb` 或拆分后的分析模块
- 权重对比 CSV
- 测试与限制说明

---

## 9. P7：提前准备数据、训练和校准接口

**优先级：中。依赖：P2。使用合成数据验收。**

### 9.1 数据协议

训练模型头的最小字段：

```text
id,text,label
```

未来形成金融情绪因子需要扩展：

```text
id,text,label,entity_id,published_at,source
```

创建 `metadata/data_schema.json`，规定：

- `id` 唯一；
- `text` 非空字符串；
- `label` 必须在 label mapping 中；
- `published_at` 带时区；
- `entity_id` 使用稳定证券/主体标识；
- 重复 ID、缺失值、非法时间的处理方式。

### 9.2 数据加载器

- 支持 CSV 和 JSONL；
- tokenizer 参数全部来自配置；
- 支持动态 padding；
- 保留样本 ID；
- split 接口支持指定 seed；
- 明确 train/dev/test 不重叠检查；
- 当前使用合成文本和虚拟标签验证管线。

### 9.3 特征缓存训练

未来冻结 backbone 时优先：

1. 一次前向缓存 12 层 pooled feature；
2. 针对每层特征训练独立 head；
3. optimizer 只接收 head 参数；
4. 每层采用相同 split、训练预算和超参数搜索空间。

训练脚本当前用合成数据只需证明：

- loss 能反向传播；
- head 参数发生变化；
- backbone 参数完全不变；
- checkpoint 可以保存和重新加载；
- 12 个 head 的指标输出格式完整。

合成数据结果不得提交为模型表现结论。

### 9.4 校准和阈值搜索接口

提前实现函数签名：

- 每个出口温度缩放；
- NLL 和 ECE；
- 最大概率、熵、margin、patience 分数；
- 在质量下降约束下搜索阈值；
- dev/calibration 与 test 严格分离。

当前只用合成 logits 做数值和控制流测试，不输出推荐阈值。

### P7 验收

- [ ] CSV/JSONL 合成数据均能完整加载。
- [ ] 重复 ID、非法 label、空文本、无时区时间会被检测。
- [ ] 合成训练一步后只有 head 参数变化。
- [ ] 保存并重载后 logits 一致。
- [ ] 温度始终为正且校准函数无 NaN/Inf。
- [ ] 阈值搜索 API 强制要求独立 calibration 输入。
- [ ] 输出明确标记 `synthetic_only=true`。

### P7 交付物

- `src/data.py`
- 数据 schema
- 只训练 heads 的训练入口
- calibration/threshold 接口
- `scripts/run_synthetic_pipeline.py`
- 对应单元测试

---

## 10. P8：情绪因子输出协议与聚合骨架

**优先级：中。依赖：P1、P4。**

### 10.1 文本级原始输出

标签映射未知时输出：

```text
text_id
entity_id
published_at
source
class_0_logit
class_1_logit
class_0_prob
class_1_prob
logit_margin_1_minus_0
max_probability
entropy
exit_layer
head_type
pooling
model_hash
```

标签映射确认后才增加：

```text
p_negative
p_positive
sentiment_score = p_positive - p_negative
```

### 10.2 因子聚合接口

只搭建函数与配置，不确定最终参数：

- 近重复新闻过滤；
- 文本到证券/主体映射；
- 发布时间统一到研究时区；
- 交易日和盘前/盘中/盘后归属；
- 来源权重；
- 时间衰减；
- 低置信度拒识；
- 同主体同时间窗聚合；
- winsorization、横截面标准化；
- 缺失值规则；
- point-in-time 检查，防止未来信息泄漏。

### 10.3 推荐的中间数据层

保持三层数据，避免覆盖原始结果：

1. `raw_predictions`：逐文本原始 logits/probabilities；
2. `mapped_predictions`：增加标签含义、主体和交易时间；
3. `daily_factor`：主体—交易日级聚合因子。

所有中间表必须保存 model hash 和聚合配置版本。

### 10.4 当前可做的合成验收

使用合成预测检查：

- 同日同主体聚合正确；
- 时间衰减方向正确；
- 重复文本不会重复计权；
- 盘后文本不会进入当天收盘前因子；
- 缺失主体或时间按规则拒绝/隔离；
- 任何转换均保留可追溯的 text ID。

### P8 验收

- [ ] 标签未知时不会输出正负情绪字段。
- [ ] 所有时间字段带时区并通过 point-in-time 检查。
- [ ] 原始输出不可变，聚合生成新表。
- [ ] 同一输入和配置版本产生一致因子结果。
- [ ] 结果可追溯到每条源文本和模型版本。
- [ ] 合成测试覆盖去重、时间边界、衰减、聚合和缺失值。

### P8 交付物

- `src/factor.py`
- 因子 schema
- 聚合配置模板
- 合成样例输出
- 对应单元测试

---

## 11. 自动测试总表

以下测试是本阶段的最低测试集：

| 测试 | 验证内容 | 通过标准 |
|---|---|---|
| `test_checkpoint_identity` | 相同权重自比较 | delta=0，cos≈1 |
| `test_key_unwrap` | 包装键/前缀处理 | 目标键集合完全一致 |
| `test_strict_backbone_load` | backbone 严格加载 | missing/unexpected/shape mismatch 均为 0 |
| `test_pooling_shapes` | 三种 pooling | 输出 `[batch, 768]` |
| `test_masked_mean_padding` | mean pooling mask | 增加右 padding 后不变 |
| `test_layer_mapping` | hidden/encoder 编号 | 1→0，12→11 |
| `test_head_initialization` | 复制头 | 初始参数逐元素等于原 fc |
| `test_random_head_seed` | 随机头复现 | 同 seed 权重一致 |
| `test_backbone_frozen` | 冻结保证 | backward 后 backbone grad 全 None |
| `test_layer_output_shape` | 多头输出 | `[batch, 12, 2]` |
| `test_probability_validity` | 概率 | 有限值、范围有效、和≈1 |
| `test_final_exit_equivalence` | 最后层一致 | Early-Exit 与完整模型 logits 接近 |
| `test_forced_exit_stops_compute` | 真实停止 | 后续层 call count=0 |
| `test_last_layer_fallback` | 阈值兜底 | 所有样本均有输出 |
| `test_active_batch_order` | 动态 batch | 输出恢复原样本顺序 |
| `test_cache_version_guard` | 缓存版本 | hash/pooling 不同则拒绝 |
| `test_synthetic_train_heads_only` | 合成训练 | 仅 head 参数变化 |
| `test_factor_time_boundary` | 时间防泄漏 | 盘后文本不进入更早因子 |
| `test_factor_traceability` | 可追溯性 | 因子可回溯至 text ID |

建议建立一个 CPU 快速测试集；CUDA 测试用 marker 单独运行。

---

## 12. 执行顺序、依赖与里程碑

### 12.1 推荐顺序

```text
P0 资产与加载
 └─ P1 完整推理候选
     ├─ P2 逐层模型头
     │   ├─ P3 逐层输出/缓存
     │   ├─ P4 真 Early-Exit
     │   │   └─ P5 性能基准
     │   └─ P7 数据/训练/校准接口
     └─ P8 情绪因子协议

P6 现有分析修正可在 P0 后并行推进
```

### 12.2 里程碑

| 里程碑 | 范围 | 完成标志 | 预计工作量 |
|---|---|---|---:|
| M0 | P0 | 模型资产可审计、backbone 严格加载 | 0.5–1 天 |
| M1 | P1 | 三种 pooling 的完整推理候选可运行 | 0.5–1 天 |
| M2 | P2–P3 | 四组 head 与逐层输出/缓存完成 | 1–2 天 |
| M3 | P4 | 最后层一致、浅层确实停止计算 | 1–2 天 |
| M4 | P5 | 固定层性能矩阵完成 | 0.5–1.5 天 |
| M5 | P6 | 现有分析的统计和结论风险修复 | 1–2 天 |
| M6 | P7–P8 | 合成数据训练、校准和因子接口打通 | 1–2 天 |
| M7 | 全部 | 报告、测试和复现命令齐全 | 0.5–1 天 |

### 12.3 每日建议

第一天：P0、P1，明确 checkpoint 能加载什么、哪些仍未知。

第二天：P2、P3，完成四组 heads 和统一逐层输出。

第三天：P4，完成真正逐层执行和最后层一致性测试。

第四天：P5、P6，跑固定层性能基准并修正分析统计。

第五天：P7、P8，用合成数据打通未来训练、校准和因子协议。

---

## 13. 数据到位后的解锁清单

以下任务当前只准备接口，必须等真实数据和任务定义到位：

### 13.1 恢复任务契约

- [ ] 找到训练脚本或原模型类。
- [ ] 确认 pooling、dropout、LayerNorm 和分类头前向。
- [ ] 确认 label mapping。
- [ ] 确认 tokenizer/max length/预处理。
- [ ] 确认原训练底座 ID 和 revision。
- [ ] 固定无泄漏的 train/dev/calibration/test split。

### 13.2 完整模型基线

- [ ] 复现完整模型 logits。
- [ ] 报告 Accuracy、Macro-F1、每类 recall、NLL、ECE。
- [ ] 保存逐样本 logits 和错误案例。

### 13.3 逐层头训练

- [ ] backbone 全冻结。
- [ ] 每层采用相同 split、预算和 seed。
- [ ] 比较复制初始化与随机初始化。
- [ ] 比较 Linear 与 LayerNorm+Linear。
- [ ] 至少运行 3 个 seed。
- [ ] 定义最浅可用层：相对最终层指标下降不超过 0.5/1/2 个百分点。

### 13.4 校准和 Early-Exit

- [ ] 每出口单独温度缩放。
- [ ] 只在 calibration/dev 上选择阈值。
- [ ] test 只作最终一次评估。
- [ ] 与固定深度退出比较。
- [ ] 输出质量—平均层数和质量—实测延迟 Pareto 曲线。

### 13.5 因子验证

- [ ] 确认正负标签后生成文本级 sentiment score。
- [ ] 在金融域标注数据上验证，而非只用通用情感数据。
- [ ] 对主体映射、时间归属和重复新闻做人工抽查。
- [ ] 做 point-in-time 回测，禁止未来信息。
- [ ] 报告因子覆盖率、稳定性、分组收益和换手等研究指标。

---

## 14. 最终验收报告模板

在 `reports/no_data_stage_report.md` 中逐项记录：

### 14.1 资产

- checkpoint/base/tokenizer/config hash；
- 环境版本；
- 参数覆盖率；
- missing/unexpected/mismatch。

### 14.2 模型前向

- 已实现 pooling；
- 当前选用 pooling 是否得到原训练证据确认；
- `fc.weight`/`fc.bias` 状态；
- 重复推理和 padding invariance 结果。

### 14.3 Heads

- head 类型、数量、参数量、初始化；
- backbone 冻结证明；
- `[batch, 12, 2]` 输出证明；
- 未训练 heads 的限制声明。

### 14.4 Early-Exit

- 最后层等价误差；
- 各固定层 call count；
- 兜底和 batch 顺序测试；
- 动态阈值仍未校准的声明。

### 14.5 性能

- 硬件、batch、长度、dtype；
- p50/p95 latency、吞吐、显存；
- 理论层数节省与实测速度差距。

### 14.6 待数据项

- label mapping；
- 原 pooling/模型类；
- 原任务数据和 split；
- 正式训练、校准和因子验证。

### 14.7 总验收签字项

- [ ] 工程链路完成。
- [ ] 自动测试全部通过。
- [ ] 无数据阶段没有越界结论。
- [ ] 数据到位后的入口、配置和命令明确。

---

## 15. 立即开工清单

按以下顺序逐项勾选：

1. [ ] 建立目录、配置和环境文件。
2. [ ] 实现 `unwrap_state_dict()` 和前缀清理。
3. [ ] 生成模型资产 manifest 和参数匹配报告。
4. [ ] 实现 CLS、pooler、masked mean 三种 pooling。
5. [ ] 加载原始 `fc`，建立完整二分类推理候选。
6. [ ] 建立固定 smoke-test 文本和 padding/batch 一致性测试。
7. [ ] 实现原始最终头与共享冻结头。
8. [ ] 实现12个复制头与12个随机对照头。
9. [ ] 验证 backbone 冻结和合成反向传播。
10. [ ] 导出 `[batch, 12, 2]` 逐层结果。
11. [ ] 实现带 call counter 的真正逐层 Early-Exit。
12. [ ] 验证最后层等价和强制浅层停止。
13. [ ] 跑固定退出层 latency/throughput 矩阵。
14. [ ] 修复 `compare_layers.py` 解包和聚合。
15. [ ] 修正 `check.ipynb` 的 mask、层编号和结论措辞。
16. [ ] 建立数据 schema、加载器和特征缓存接口。
17. [ ] 用合成数据验证只训练 heads 的管线。
18. [ ] 建立温度校准和阈值搜索接口，不产生正式阈值。
19. [ ] 建立文本级预测和主体—日期因子聚合协议。
20. [ ] 运行全部 CPU 测试和可用的 CUDA smoke test。
21. [ ] 完成 `no_data_stage_report.md`。
22. [ ] 冻结本阶段配置和产物版本，等待真实数据解锁正式实验。

执行时始终遵循一个原则：**无数据阶段把工程做完整、把假设写清楚、把测试做严格，但不把未验证输出包装成模型能力或投资因子结论。**
