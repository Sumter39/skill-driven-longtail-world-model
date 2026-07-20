# 测试目录

根目录的 `conftest.py` 提供共享fixture，其余测试按职责分组：

| 目录 | 职责 |
|---|---|
| `unit/core/` | 配置和公共Schema |
| `unit/data/` | AV2读取、坐标变换和场景清单 |
| `unit/skills/` | 技能YAML、注册表、几何和30类规则检测 |
| `unit/seeds/` | 候选记录、参数采样和确定性筛选 |
| `unit/visualization/` | BEV和审核图逻辑 |
| `workflows/data/` | 下载、Train池划分和完整性验证命令 |
| `workflows/seed_detection/` | 候选扫描、审核渲染和正式种子选择流程 |

运行全部测试：

```bash
uv run pytest -q
```

按职责运行：

```bash
uv run pytest tests/unit/skills -q
uv run pytest tests/workflows/seed_detection -q
```
