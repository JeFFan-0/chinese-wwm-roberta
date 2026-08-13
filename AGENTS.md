# Agent 工作规范（AGENTS）

> 给在这个仓库里工作的 AI Agent / 协作者看的硬性约定。核心原则：**这个项目当前没有标签数据，任何超出“工程正确性”的结论都是越界。**

## 铁律（禁止越界声明）

不得声称：
- 某一层分类效果最好 / 已包含足够情绪信息；
- 已选定正式 Early-Exit 阈值；
- 模型 Accuracy / Macro-F1 / 因子预测能力；
- 把 `class 0/1` 命名为 negative/positive（标签映射未知）；
- 把低置信度解释为“中性”（二分类没有中性类）；
- 把“固定层计算量减少”等同于“真实部署加速”。

合成数据产出的任何结果一律标记 `synthetic_only=true`，并注明“合成数据结果不提交为模型表现结论”。

## 已知边界

- pooling 未确认（候选 cls / pooler / masked_mean），`configs/model.yaml` 里 `pooling: unknown`。
- 标签映射未知，输出字段固定用 `class_0_* / class_1_*`。
- backbone 恒冻结，只允许训练 head。
- 层编号规范：encoder layer 0–11；hidden index = encoder_layer + 1（hidden index 0 是 embedding，不进 Early-Exit）。

## 环境与运行

```bash
conda activate nlp_fjq          # torch 2.1.2 / transformers 4.51.3
python -m pytest tests/ -q       # 141 个用例，全部应通过
```

- 单测是 CPU 的；CUDA 相关有 marker 待补，别在单测里硬跑 GPU。
- 路径不要写死机器绝对路径：用 `configs/model.yaml` 里的 `${ENV:-default}` 覆盖。

## 代码结构约定

- 模型核心 → `src/models/`；逐层头/探针 → `src/probes/`；数据 → `src/data/`；
  因子 → `src/factors/`；评估 → `src/evaluation/`；共享配置加载 → `src/config.py`。
- 可运行的入口脚本放 `scripts/`，不要在 `src/` 里放 `__main__`。
- 新脚本一律用 `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 把仓库根加进 `sys.path` 再 `import src.*`。
- 改动任何模块后，跑一遍 `python -m pytest tests/ -q` 确认没破坏 141 个测试。

## Git 约定

- 大模型权重、`artifacts/`、`src/data/fetch/config.py`（含凭据）一律不提交（见 `.gitignore`）。
- 提交前确认没有把凭据或大文件带进去；commit message 用中文描述做了什么。
- 未提交改动（如 `data_fetch` 取数脚本）在提交前先和作者确认。

## 文档更新约定

- 改了文件结构 → 同步更新 `PROJECT_MAP.md`；
- 改了进度/结论 → 同步更新 `CURRENT_STATE.md`；
- 新增实验 → 补到 `docs/EXPERIMENTS.md` 和 `reports/`。
