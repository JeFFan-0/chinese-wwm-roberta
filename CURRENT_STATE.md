# 当前状态（CURRENT_STATE）

> 用途：不知道项目做到哪一步了，就看这里。一句话：**无数据阶段（P0–P8）已全部完成，141 个自动测试通过，等真实数据到位后进入“只训头 + 校准 + 评估”阶段。**

## 一句话定位

这是一个**没有原任务数据**的中文 RoBERTa 工程。手上只有：
1. 一个微调好的 checkpoint（`chinese-wwm-roberta.ckpt`）；
2. 本地预训练底座（`chinese-roberta-wwm-ext/`）。

总目标是在**不改 BERT 权重、不依赖标签**的前提下，把“可信加载 → 推理 → 逐层头 → Early-Exit → 基准 → 数据接口 → 因子协议”这套基础工程全部搭好并测通。数据到位后只训模型头、校准阈值、做评估即可。

## 完成状态（P0–P8，2026-08-12）

| 阶段 | 交付 | 验证要点 |
|---|---|---|
| P0 资产与可信加载 | `src/models/checkpoint.py`、`scripts/verify_assets.py` | backbone 严格加载 199/199，覆盖率 100% |
| P1 完整推理候选 | `src/models/pooling.py`、`modeling.py` | 三 pooling 输出 `[B,768]`，概率行和≈1 |
| P2 逐层模型头 | `src/probes/heads.py` | Head A–E，backbone 冻结，`[B,12,2]` |
| P3 逐层输出/缓存 | `src/probes/layer_outputs.py` | 统一 schema，缓存版本保护 |
| P4 真 Early-Exit | `src/models/early_exit.py` | layer 11 = 完整前向，退出后不再调用后续层 |
| P5 性能基准 | `scripts/benchmark_fixed_exit.py` | A100/FP32 全矩阵 |
| P6 分析修正 | `src/evaluation/analysis.py` | 真实 max abs delta = 0.001488 |
| P7 数据/训练/校准 | `src/data/dataset.py`、`training.py`、`calibration.py` | 只训 head（backbone 不变），温度>0 |
| P8 因子协议 | `src/factors/factor.py` | 三层表、point-in-time |

## 已经确定的事实

- backbone 是 BERT：12 encoder layer、hidden 768、12 head、intermediate 3072、vocab 21128、激活 GELU。
- checkpoint 含 199 个 `bert.*` 键 + `fc.weight [2,768]` + `fc.bias [2]`（二分类头候选）。
- 微调对 backbone 改动极小：逐层权重最大绝对误差约 **1.5e-3**；激活差异随层深单调放大（cos layer0=0.99998 → layer12=0.980）。
- Early-Exit 引擎真正逐层执行；固定 layer 11 与完整前向数值一致。

## 仍然未知（禁止提前断言）

- `class 0` / `class 1` 分别代表什么（不能命名 positive/negative）。
- 原模型头用哪种 pooling（cls / pooler / masked_mean 未确认）。
- 原分类头前是否还有 dropout / LayerNorm / 激活层。
- 原 tokenizer 参数、最大长度、清洗规则；底座是否就是微调时用的 revision。
- 模型真实 Accuracy / Macro-F1 / 校准性 / 因子有效性（没有标签，测不了）。

## 数据到位后的下一步（按顺序）

1. 确认 label mapping 与 pooling，复现完整模型基线。
2. 用真实数据训练每层头，定义“最浅可用层”。
3. 每个出口做温度校准，选阈值（只用 calibration/dev，test 严格隔离）。
4. 金融域因子验证（point-in-time 回测）。

## 取数（数据获取）现状

- `src/data/fetch/` 有 5 个数据源的取数脚本（公司公告 / 调研问答 / 新闻 / 研报 / 社交文本）。
- 已可连通：Wind Oracle、datayes MySQL（新闻/快讯）、研报 zyyx2 Oracle。
- 暂不可用：社交文本 FTP（DNS 解析失败/网关拦截）、新闻正文 S3（凭据未知）。
- 凭据在 `src/data/fetch/config.py`（gitignore，禁止提交）。

## 环境

- conda env：`nlp_fjq`（Python 3.10，torch 2.1.2 / transformers 4.51.3，机器有 CUDA）。
- 全部脚本/测试/取数统一在 `nlp_fjq` 下执行。
- ⚠ `requirements.txt` 锁定的版本是 torch 2.5.1 / transformers 5.15.0，与当前 `nlp_fjq` 实际安装（2.1.2 / 4.51.3）不一致——版本对齐留待后续处理。
