# chinese-wwm-roberta

中文 RoBERTa（`hfl/chinese-roberta-wwm-ext` 底座）微调模型的无数据阶段工程。
阶段目标：**不修改 BERT 各层权重、不依赖标签数据**，完成可信加载、完整推理候选、
逐层模型头、真实 Early-Exit、性能基准、未来数据接口与情绪因子输出协议。

> 状态：无数据阶段完成（P0–P8），140 个自动测试通过。验收报告：
> [`reports/no_data_stage_report.md`](reports/no_data_stage_report.md)。
> 执行方案：`TODO.md`；阶段计划：`plan.md`。

## 资产

- `chinese-wwm-roberta.ckpt`：微调 checkpoint（199 个 `bert.*` backbone 键 + `fc.weight [2,768]` + `fc.bias [2]`）。
- `chinese-roberta-wwm-ext/`：本地预训练底座（已附加 `model.safetensors` 安全版权重）。
- 全部哈希/大小/结构清单：`metadata/model_manifest.json`（`scripts/verify_assets.py` 生成）。

## 工程结构

```text
configs/      model / heads / early_exit / benchmark / factor 配置
metadata/     model_manifest.json, data_schema.json, factor_schema.json
src/          checkpoint, modeling, pooling, heads, layer_outputs,
              early_exit, analysis, data, training, calibration, factor, config
scripts/      verify_assets / smoke_inference / report_heads / export_layer_outputs /
              demo_early_exit / benchmark_fixed_exit / run_synthetic_pipeline / demo_factor
tests/        140 个 pytest 用例（CPU；CUDA 有 marker 待补）
reports/      no_data_stage_report.md, tables/, figures/
artifacts/    缓存与大产物（gitignore，不提交）
```

## 快速开始

```bash
conda activate 26intern          # torch 2.5.1 / transformers 5.15.0 / CUDA A100
pip install pytest                # 若未装

python scripts/verify_assets.py              # P0 资产核验（含 SHA-256，~30s）
python scripts/smoke_inference.py --device cpu
python scripts/demo_early_exit.py --device cpu
python scripts/benchmark_fixed_exit.py --device cuda
python scripts/run_synthetic_pipeline.py --device cpu
python -m pytest tests/ -q
```

## 关键结论（严格限制，见报告 §14）

- backbone 与底座严格对齐：matched=199、missing/unexpected/shape_mismatch=0、参数覆盖率 100%。
- 原分类头 `fc` 存在但 **pooling 未确认**（候选：cls / pooler / masked_mean）；
  标签映射未知，输出固定为 `class_0/1`，不命名为 positive/negative。
- Early-Exit 引擎真正逐层执行并停止后续层；固定 layer 11 与完整前向完全一致。
- 性能矩阵（A100/FP32）仅表述为"若未来证明可在 layer k 退出则测得速度 X"。
- 数据/训练/校准/因子接口已用合成数据验证（全部 `synthetic_only=true`），等待真实数据解锁正式实验。

## 数据到位后的下一步

见 `reports/no_data_stage_report.md` §14.6 与 `TODO.md` §13：
确认 label mapping 与 pooling → 复现完整模型基线 → 每层头训练与最浅可用层定义 →
每出口温度校准与阈值选择（仅 calibration/dev）→ 金融域因子验证（point-in-time 回测）。
