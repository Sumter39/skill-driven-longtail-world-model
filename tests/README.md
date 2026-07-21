# 测试目录

根目录的 `conftest.py` 提供共享fixture，其余测试按职责分组：

| 目录 | 职责 |
|---|---|
| `unit/core/` | 配置和公共Schema |
| `unit/data/` | AV2读取、坐标变换和场景清单 |
| `unit/models/` | 条件CVAE前向传播、先验采样、掩码语义、可复现性和梯度回传 |
| `unit/skills/` | 34类正式技能、5类候选规则、状态分区、YAML、注册表、几何和规则检测 |
| `unit/seeds/` | 候选记录、参数采样和确定性筛选 |
| `unit/training/` | CVAE训练配置、指标、检查点、损失、AMP、优化步骤和性能基准逻辑 |
| `unit/visualization/` | BEV和审核图逻辑 |
| `workflows/data/` | 下载、Train池划分和完整性验证命令 |
| `workflows/modeling/` | CVAE数据缓存、训练与断点恢复、评估诊断和GPU冒烟流程 |
| `workflows/seed_detection/` | 候选扫描、审核渲染和正式种子选择流程 |

运行全部测试：

```bash
./.venv/bin/python -m pytest -q
```

按职责运行：

```bash
./.venv/bin/python -m pytest tests/unit/skills -q
./.venv/bin/python -m pytest tests/workflows/seed_detection -q
```
