# 无数据阶段验收报告（chinese-wwm-roberta）

> 生成日期：2026-08-12
> 阶段定义见 `TODO.md`：不修改 BERT 各层权重、不依赖标签数据，完成可信加载、完整推理、
> 逐层模型头、真实 Early-Exit、性能基准、未来数据接口与情绪因子输出协议。
> 所有数值均为本机（A100 80GB / torch 2.5.1 / transformers 5.15.0 / FP32）实测。

---

## 14.1 资产

| 资产 | 路径 | SHA-256（前 16） | 大小 |
|---|---|---|---|
| 微调 checkpoint | `chinese-wwm-roberta.ckpt` | `b6de21972c150139…` | 409,126,964 B |
| 底座权重 | `chinese-roberta-wwm-ext/pytorch_model.bin` | `1ded5a5a1c7841de…` | 411,578,458 B |
| 底座 config | `chinese-roberta-wwm-ext/config.json` | `61609babfbada201…` | 689 B |
| 底座 tokenizer | `chinese-roberta-wwm-ext/tokenizer.json` | `53ff61207898738b…` | 268,961 B |
| 底座权重（安全版） | `chinese-roberta-wwm-ext/model.safetensors` | 转换产物 | 476.5 MB |

- 完整清单：`metadata/model_manifest.json`。
- 环境：Python 3.11.14 / torch 2.5.1 / transformers 5.15.0 / CUDA 11.8 / cuDNN 90100 / NVIDIA A100 80GB PCIe。
- 结构：BERT，12 encoder layer，hidden 768，12 head，intermediate 3072，vocab 21128，激活 GELU。
- **backbone 严格加载**：matched=199，missing=0，unexpected=0，shape_mismatch=0，
  tensor 覆盖率 100%，参数覆盖率 100%（`reports/tables/checkpoint_key_report.*`）。
- ckpt 相对底座全量：matched=199（`bert.*` backbone），only-base=8（`cls.*` MLM/NSP 头），
  only-ckpt=2（`fc.*` 分类头）。

## 14.2 模型前向

- 已实现三种 pooling：`cls` / `pooler` / `masked_mean`（`src/pooling.py`）。
- **当前选用 pooling 是否得到原训练证据确认：否**（`pooling_confirmed=false`，配置与日志均显式标记）。
- `fc.weight` 存在，shape `[2, 768]`，numel 1536；`fc.bias` 存在，shape `[2]`。
- 重复运行 logits 完全一致；单条推理与组批结果最大差 ~6e-7；
  右侧 padding 不改变 CLS/masked-mean 结果；空白文本不产生 NaN。
- smoke 输出：`reports/tables/smoke_inference.csv`（示例见 `scripts/smoke_inference.py`）。
- 说明：不同 pooling 得到的概率分布差异显著（如"好"在 cls 下 class_0 概率 0.81、
  pooler 下 0.40），因此当前**任何** pooling 结果都只是候选，不能用于语义结论。

## 14.3 Heads

| Head | 类型 | 初始化 | 参数量 | 可训练 |
|---|---|---|---|---|
| A `original_final_head` | Linear | 复制自 checkpoint fc | 1,538 | 否（只读） |
| B `shared_frozen_head` | Linear（共享） | 复制自 fc | 1,538 | 否 |
| C `copied_layer_heads` | 12× Linear | 复制自 fc | 18,456 | 是（未来） |
| D `random_layer_heads` | 12× Linear | 固定 seed 42 随机 | 18,456 | 是（对照） |
| E `normalized_layer_heads` | LayerNorm+Dropout+Linear | 复制自 fc | 36,888 | 是（非默认） |

- backbone 冻结证明：所有 `bert.*` 参数 `requires_grad=False`；合成反向传播后 backbone
  gradient 全为 `None`，可训练 head gradient 非零（`tests/test_heads.py`）。
- 生产层头输出 `[batch, 12, 2]`；Head A 输出 `[batch, 2]`（最后层基线）。
- 共享头与复制头初始状态在 12 层输出完全一致；随机头可由 seed 精确复现。
- 层编号规范：hidden index 0 = embedding（仅诊断），encoder layer 0-11；**无 "encoder layer 12"**。
- **限制声明**：Head C/D/E 均未训练，其输出没有任何语义含义，不得解释为层质量或情绪能力。

## 14.4 Early-Exit

- 引擎 `src/early_exit.py` 真正逐层执行：`embeddings` → `create_bidirectional_mask` →
  循环 `encoder.layer[i]` → 候选层 pooling + head → 退出判定，**退出后不再调用后续层**。
- 固定 layer 11 退出与普通完整前向 logits **完全一致（diff=0.0）**。
- 固定 layer k 退出时，层 k+1..11 的 call counter 全为 0；`executed_layer_count == k+1`
  （hook 独立计数，见 `scripts/demo_early_exit.py` 追踪日志）。
- 极高阈值 → 全部由最后层兜底；极低阈值 → 首个候选层退出；active-set 输出恢复原顺序。
- **动态阈值未校准**：max_prob/margin 退出仅用于控制流 smoke test，不用于正式部署。
- 生产评估 head 只在退出层计算（`evaluate_layers`），不包含无关的中间层头开销。

## 14.5 性能

- 硬件/软件/配置见 `reports/tables/benchmark_env.json`；全矩阵 112 行见
  `reports/tables/fixed_exit_benchmark.csv`。
- 矩阵：seq_len {32,64,128,256} × batch {1,8,16,32} × 固定退出层 {2,4,6,8,10,11}
  + 完整 12 层基线；warmup 10 次，正式 50 次，GPU 同步计时，固定 seed 输入，FP32。
- 代表配置（len=128, batch=16）p50 与速度比（基线 = 同配置 engine 完整 12 层）：

| 退出层 | p50 (ms) | samples/s | 实测加速 | 理论 12/(k+1) | 差距 |
|---:|---:|---:|---:|---:|---:|
| 2 | 7.19 | 2224 | 3.90× | 6.00× | 2.10 |
| 4 | 11.78 | 1359 | 2.38× | 3.00× | 0.62 |
| 6 | 16.38 | 977 | 1.71× | 2.00× | 0.29 |
| 8 | 21.01 | 761 | 1.34× | 1.50× | 0.16 |
| 10 | 25.70 | 623 | 1.09× | 1.20× | 0.11 |
| 11 | 28.07 | 570 | 1.00× | 1.09× | 0.09 |

- 完整 12 层 p50 范围：7.69–107.58 ms（随长度/batch）。完整基线本身含逐层 pooling 与
  全部 head 的评估开销，故浅层退出实测加速**低于**纯层数比，这正是理论节省与实测的差距来源。
- 结论表述严格限制为：**"若未来证明可在 layer k 退出，则在本硬件/批/长度下测得速度 X"**，
  不代表 layer k 已足够完成情绪任务。
- 图：`reports/figures/latency_exit_layer.png`、`reports/figures/speedup_vs_theory.png`。

## 14.6 待数据项（数据到位后解锁）

- [ ] label mapping（class 0/1 含义），确认 pooling / 原模型类 / 前向结构；
- [ ] 原 tokenizer 参数、max length、截断方向与数据清洗规则；
- [ ] 原训练底座 ID 与 revision；
- [ ] 无泄漏的 train/dev/calibration/test split；
- [ ] 完整模型基线（Accuracy、Macro-F1、每类 recall、NLL、ECE、逐样本 logits）；
- [ ] 每层头训练（复制 vs 随机、Linear vs LayerNorm+Linear、≥3 seed、最浅可用层定义）；
- [ ] 每出口温度校准 + 仅 calibration/dev 上选阈值 + test 单次评估 + Pareto 曲线；
- [ ] 金融域标注数据上的因子验证（主体映射、时间归属、point-in-time 回测）。

## 14.7 总验收签字项

- [x] 工程链路完成（P0–P8 全部实现，140 个自动测试全部通过）；
- [x] 无数据阶段没有越界结论（未知标签仅以 class_0/1 输出；未声称任何层质量/情绪能力；
      未选正式 Early-Exit 阈值；未报告 Accuracy/Macro-F1）；
- [x] 数据到位后的入口、配置和命令明确（`scripts/run_synthetic_pipeline.py`、
      `scripts/benchmark_fixed_exit.py`、`scripts/export_layer_outputs.py`、
      `src/data.py` / `src/training.py` / `src/calibration.py` / `src/factor.py`）。

## 复现命令

```bash
conda activate 26intern
python scripts/verify_assets.py                 # P0 资产与严格加载（含哈希）
python scripts/smoke_inference.py --device cpu  # P1 三种 pooling smoke
python scripts/report_heads.py                  # P2 heads 参数清单
python scripts/export_layer_outputs.py --device cpu  # P3 逐层导出 + 缓存
python scripts/demo_early_exit.py --device cpu  # P4 Early-Exit 追踪日志
python scripts/benchmark_fixed_exit.py --device cuda  # P5 固定层性能矩阵
python compare_layers.py --device cpu           # P6 修正后的逐层对比
python scripts/run_synthetic_pipeline.py --device cpu  # P7 合成数据管线
python scripts/demo_factor.py                   # P8 因子三层管线演示
python -m pytest tests/ -q                     # 全部自动测试
```
