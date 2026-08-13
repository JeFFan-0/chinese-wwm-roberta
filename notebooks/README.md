# notebooks/ 说明

当前只有**一个** notebook：`check.ipynb`，是早期探索工作簿（逐层权重/激活对比），
其核心逻辑已重写进 `src/evaluation/analysis.py` 与 `scripts/compare_layers.py`，notebook 保留作记录。

规划中的 notebook 分工（等真实数据到位后再拆分/新建）：

| 规划文件 | 对应能力 | 当前等价入口 |
|---|---|---|
| `01_data_check.ipynb` | 数据校验/切分 | `src/data/dataset.py` + `scripts/run_synthetic_pipeline.py` |
| `02_layer_analysis.ipynb` | 逐层分析 | `scripts/compare_layers.py` / `scripts/export_layer_outputs.py` |
| `03_head_analysis.ipynb` | 逐层头分析 | `scripts/report_heads.py` / `scripts/demo_early_exit.py` |

> 说明：`check.ipynb` 是历史记录，其内部 import 可能仍是旧的扁平路径（`from src.xxx`）。
> 如果要重跑，需把其中的 `from src.modeling/heads/...` 改成子包路径（见 `PROJECT_MAP.md`）。
