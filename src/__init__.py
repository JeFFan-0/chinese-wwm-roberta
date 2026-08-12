"""chinese-wwm-roberta 无数据阶段工程源码包。

各子模块职责：
- checkpoint: checkpoint 安全解包、前缀处理、键匹配报告
- modeling:   backbone 与完整二分类推理候选
- pooling:    CLS / pooler / masked-mean 三种 pooling
- heads:      各类逐层模型头
- layer_outputs: 逐层输出与导出格式
- early_exit: 真正逐层执行的 Early-Exit 引擎
- factor:     情绪因子输出协议与聚合接口
- data:       CSV/JSONL/合成数据加载与训练管线
"""

__version__ = "0.1.0"
