# 工作流（WORKFLOW）

> 从零到出结果的完整操作顺序。当前无数据阶段，P0–P8 都已完成；数据到位后按“下一步”走。

## 无数据阶段（已完成）的流水线

```bash
conda activate nlp_fjq

# P0 核验资产，生成 manifest
python scripts/verify_assets.py

# P1 推理 smoke test
python scripts/smoke_inference.py --device cpu

# P2 heads 清单
python scripts/report_heads.py

# P3 导出逐层输出
python scripts/export_layer_outputs.py --device cpu

# P4 Early-Exit 演示
python scripts/demo_early_exit.py --device cpu

# P5 性能基准（需要 GPU）
python scripts/benchmark_fixed_exit.py --device cuda

# P6 逐层对比
python scripts/compare_layers.py --device cpu

# P7 合成数据完整管线
python scripts/run_synthetic_pipeline.py --device cpu

# P8 因子演示
python scripts/demo_factor.py

# 回归测试
python -m pytest tests/ -q
```

## 数据到位后的流水线

1. **取数**：跑 `src/data/fetch/` 里对应数据源的脚本，把原始文本落到 `~/data/NLP/`。
2. **建数据集**：用 `src/data/dataset.py` 的 `load_dataset` / `validate_dataset` / `split_dataset` 生成 id/text/label（+entity_id/published_at/source）。
3. **确认 label mapping 与 pooling**：这是所有下游工作的前提（现在都是 unknown）。
4. **逐层训 head**：`src/probes/training.py`（一次前向缓存 12 层特征 → 逐层独立 head）。
5. **逐层分析选“最浅可用层”**：`src/evaluation/analysis.py` + `scripts/compare_layers.py`。
6. **校准 + 阈值**：`src/evaluation/calibration.py`（只用 calibration/dev，test 隔离）。
7. **因子聚合**：`src/factors/factor.py`（point-in-time 回测）。

## 约定

- 每个脚本 `--help` 有用法；脚本 docstring 写明了它输出到 `reports/` 的哪个文件。
- 所有路径相对仓库根，可用 `configs/*.yaml` + `${ENV:-default}` 覆盖。
- 产出物：中间表 → `reports/tables/`，图 → `reports/figures/`，缓存/大产物 → `artifacts/`。
