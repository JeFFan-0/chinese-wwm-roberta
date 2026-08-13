# 项目文件地图（PROJECT_MAP）

> 用途：不知道某个文件是干什么的，就查这张表。目录树按“读代码入口 → 模型 → 探针 → 数据 → 因子 → 评估 → 脚本 → 配置 → 测试 → 产物”的顺序排列。

## 顶层（先看这些）

| 文件 | 作用 |
|---|---|
| `README.md` | 项目总览：目标、结构、快速开始、关键结论 |
| `PROJECT_MAP.md` | 本文件：每个文件/目录的说明 |
| `CURRENT_STATE.md` | 项目进度（P0–P8 完成情况 + 下一步） |
| `AGENTS.md` | Agent 工作规范（约定、禁项、提交流程） |
| `TODO.md` | 完整执行方案与验收清单（P0–P8 明细，历史文档） |
| `requirements.txt` | Python 依赖（torch 2.5.1 / transformers 5.15.0 等） |
| `.gitignore` | 忽略大模型权重、artifacts、缓存、取数凭据 |
| `conftest.py` | pytest 根配置：把仓库根加入 `sys.path`，让测试 `import src.*` |
| `chinese-wwm-roberta.ckpt` | 微调 checkpoint（约 409MB，gitignore） |
| `chinese-roberta-wwm-ext/` | 本地预训练底座 `hfl/chinese-roberta-wwm-ext`（含 `model.safetensors`） |

## src/ 源码包

### src/models/ — 模型核心

| 文件 | 作用 | 阶段 |
|---|---|---|
| `checkpoint.py` | checkpoint 安全解包、前缀处理、键匹配报告（weights_only 反序列化） | P0 |
| `modeling.py` | backbone + 完整二分类推理候选（冻结 BERT + 原 `fc`），三种 pooling 可切换 | P1 |
| `pooling.py` | CLS / pooler / masked-mean 三种 pooling 的统一接口 | P1 |
| `early_exit.py` | 真正逐层执行的 Early-Exit 引擎（每层决定是否退出，layer 11 兜底） | P4 |

### src/probes/ — 逐层模型头 / 探针

| 文件 | 作用 | 阶段 |
|---|---|---|
| `heads.py` | 逐层模型头 Head A–E（原始/共享/复制/随机/归一化），backbone 恒冻结 | P2 |
| `layer_outputs.py` | 逐层输出统一 schema、导出、pooled-feature 缓存（版本保护） | P3 |
| `training.py` | 只训练 head 的训练入口（一次前向缓存 12 层特征 → 逐层训独立 head） | P7 |

### src/data/ — 数据

| 文件 | 作用 | 阶段 |
|---|---|---|
| `dataset.py` | 数据协议与加载器：CSV/JSONL 加载、校验、split、动态 padding | P7 |
| `fetch/` | 外部数据源取数脚本（见下方“取数脚本”） | — |

### src/factors/ — 情绪因子

| 文件 | 作用 | 阶段 |
|---|---|---|
| `factor.py` | 情绪因子输出协议：raw → mapped → daily_factor 三层聚合 + point-in-time | P8 |

### src/evaluation/ — 评估

| 文件 | 作用 | 阶段 |
|---|---|---|
| `analysis.py` | 逐层对比分析（微调 vs 底座）：逐参数聚合、真实 max abs delta | P6 |
| `calibration.py` | 校准与阈值搜索接口：温度缩放、NLL/ECE、退出分数 | P7 |

### src/config.py（共享工具）

YAML 配置加载 + `${ENV:-default}` 环境变量覆盖，避免写死机器绝对路径。

## scripts/ 运行脚本

| 文件 | 作用 | 阶段 |
|---|---|---|
| `verify_assets.py` | 资产核验（SHA-256），生成 `metadata/model_manifest.json` | P0 |
| `smoke_inference.py` | 三种 pooling 的完整推理 smoke test | P1 |
| `report_heads.py` | Head A–E 初始化清单与参数量报告 | P2 |
| `export_layer_outputs.py` | 导出逐层输出 CSV/Parquet + 特征缓存 | P3 |
| `demo_early_exit.py` | Early-Exit 固定层/动态阈值演示（执行追踪日志） | P4 |
| `benchmark_fixed_exit.py` | A100/FP32 固定层性能矩阵（latency/吞吐/显存） | P5 |
| `compare_layers.py` | 微调 ckpt vs 底座逐层逐组件统计（生成 tables/*.csv） | P6 |
| `run_synthetic_pipeline.py` | 合成数据完整管线：加载→校验→split→缓存→训 head→校准 | P7 |
| `demo_factor.py` | 三层因子管线合成样例输出 | P8 |
| `convert_to_safetensors.py` | 把 .bin/.ckpt 转为 safetensors（CVE-2025-32434 规避） | 工具 |

## 取数脚本 src/data/fetch/

从外部数据源取**原始文本数据**（输出到 `~/data/NLP/...`，与核心无数据阶段解耦）。

| 文件 | 数据源 | 取什么 |
|---|---|---|
| `config.py` | — | 数据源连接凭据（**gitignore，禁止提交**） |
| `common.py` | — | 共享工具：Oracle 连接、JSONL 流式写、窗口日期 |
| `verify_connections.py` | 5 数据源 | 只读验证连接与表结构 |
| `fetch_company_filings.py` | Wind Oracle | 公司公告 ASHAREANNINF + ASHAREANNTEXT |
| `fetch_field_research.py` | Wind Oracle | 调研问答 ASHAREISQA |
| `fetch_news.py` | datayes MySQL | 新闻/微信/快讯（正文在 S3，当前落 S3_URL） |
| `fetch_research_report.py` | Oracle zyyx2 | 研报盈利预测 RPT_FORECAST_STK |
| `fetch_social_text.py` | datayes FTP | 社交文本（当前网络不可达，仅 --list-only） |

## configs/ 配置

| 文件 | 作用 |
|---|---|
| `model.yaml` | 路径、底座信息、pooling/label 占位符（路径支持 `${ENV:-default}`） |
| `heads.yaml` | Head A–E 启用与初始化策略 |
| `early_exit.yaml` | 候选退出层、兜底层、pooling、执行追踪 |
| `benchmark.yaml` | 基准矩阵（seq_len × batch × 退出层） |
| `factor.yaml` | 因子聚合骨架参数（占位，未标定） |

## metadata/ 清单与 schema

| 文件 | 作用 |
|---|---|
| `model_manifest.json` | 模型资产哈希/大小/结构清单（verify_assets 生成） |
| `data_schema.json` | 数据字段规范（id/text/label/entity_id/published_at/source） |
| `factor_schema.json` | 因子三层表 schema |

## notebooks/

| 文件 | 作用 |
|---|---|
| `check.ipynb` | 最早的探索工作簿（逐层权重/激活对比），核心逻辑已移入 src/ |
| `README.md` | 说明 notebook 规划与当前状态 |

## tests/（141 个用例）

| 文件 | 测什么 |
|---|---|
| `test_checkpoint.py` | 安全加载、前缀、键匹配 |
| `test_pooling.py` / `test_padding.py` | 三种 pooling、padding 不变性 |
| `test_heads.py` | Head A–E 结构与冻结 |
| `test_layer_outputs.py` / `test_cache_version_guard.py` | 统一 schema、缓存版本保护 |
| `test_early_exit.py` | 逐层执行、layer 11 = 完整前向 |
| `test_analysis.py` | 逐层对比统计 |
| `test_data.py` / `test_training.py` / `test_calibration.py` | 数据校验、只训 head、校准数值 |
| `test_factor_schema.py` | 因子三层表与 point-in-time |

## reports/ 与 artifacts/

- `reports/no_data_stage_report.md`：无数据阶段验收报告（主报告）。
- `reports/tables/`：各阶段产出的 CSV/Parquet（基准、逐层统计、heads 清单等）。
- `reports/figures/`：基准图表（latency/加速比）。
- `artifacts/`：缓存与大产物（pooled_features、synthetic、safetensors 等，gitignore）。
