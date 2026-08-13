# 实验记录（EXPERIMENTS）

> 已做/在做的实验、方法与结论。无数据阶段只做“工程正确性”验证，不做“模型表现”结论。主报告见 `reports/no_data_stage_report.md`。

## 已完成的实验

### P0 资产核验
- 对所有模型资产算 SHA-256、大小、结构清单 → `metadata/model_manifest.json`。
- backbone 严格加载 199/199，missing/unexpected/shape_mismatch=0，覆盖率 100%。

### P1 推理 smoke test
- 三种 pooling 输出 `[B,768]`；概率行和≈1；单条=组批；padding 不变性。

### P2 逐层头
- Head A–E 全开；backbone 冻结（grad 全 None）；随机头固定 seed 可复现；`[B,12,2]`。

### P3 逐层输出/缓存
- 行数 = 样本 ×（12 逐层头 + 1 最终头）；缓存版本保护（model_hash/pooling 不符即拒）。

### P4 Early-Exit
- layer 11 = 完整前向（diff 0）；退出后 k+1..11 call count=0；active-set 恢复顺序。

### P5 性能基准（A100/FP32）
- 矩阵：seq_len {32,64,128,256} × batch {1,8,16,32} × 退出层 {2,4,6,8,10,11} + 完整模型基线。
- 产出 `reports/tables/fixed_exit_benchmark.csv` + `reports/figures/`（latency、加速比）。
- 措辞：只能说“若未来证明可在 layer k 退出，则测得速度 X”。

### P6 逐层对比（微调 vs 底座）
- 真实 max abs delta = **0.001488**（backbone 几乎没动）。
- 逐参数聚合（`delta_l2_layer = sqrt(Σ||ΔW_p||²)`）；masked 统计；激活差异随层深放大。

### P7 数据/训练/校准
- CSV/JSONL 校验（重复 ID/空文本/非法 label/无时区时间）；split 不重叠。
- 只训 head：loss 反向传播、head 参数变、backbone 完全不变、checkpoint 可存可载。
- 校准：温度恒正、NLL/ECE 数值正确、阈值搜索强制独立 calibration 输入。

### P8 因子协议
- 三层表（raw→mapped→daily）、point-in-time、去重/衰减/缺失值、可追溯。
- 标签映射未知时**不输出**正负情绪字段。

## 待数据解锁的实验

1. 复现完整模型基线（确认 label mapping + pooling 后）。
2. 每层头训练 → “最浅可用层”定义。
3. 每出口温度校准 + 阈值选择（仅 calibration/dev）。
4. 金融域因子 point-in-time 回测。

## 历史早期探索

- `notebooks/check.ipynb` 是原始探索工作簿（逐层权重/激活对比，31 cell），其核心逻辑已重写进 `src/evaluation/analysis.py` 和 `scripts/compare_layers.py`。
- `docs/PLAN.md` 是迁移前的分析计划（历史文档，状态已过时，以 `CURRENT_STATE.md` 为准）。
