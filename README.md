# SkillDrive

**Skill-Driven Long-Tail Scenario Generation for Autonomous Driving**

SkillDrive研究如何使用技能规则和学习式轨迹世界模型，从真实驾驶片段构造可控、连续并符合交通约束的长尾场景。

当前仓库已经完成**前期准备、39个规则实现和正式候选种子检测**：环境、数据接口、规则执行、BEV审核、20,000场景正式扫描和5,000场景确定性筛选均已完成。扫描后最终定稿为34类正式技能与5类零命中候选规则；尚未进入模型训练、批量反事实生成或下游正式实验。

## 技术路线

```text
Argoverse 2 Motion Forecasting
        ↓
矢量地图与多智能体历史轨迹
        ↓
技能规则条件
        ↓
条件CVAE轨迹世界模型（后续阶段）
        ↓
运动学、地图、交通规则和风险过滤
        ↓
BEV审核与轨迹预测增强验证
```

BEV用于统一的俯视场景表达和可视化，不作为像素生成模型。最终技能体系已经由用户确认：六个技能族中包含34类正式技能，其中A类14个、B类20个，族规模为4、4、6、6、7、7；另保留5类本轮零命中的候选规则。正式与候选合计39份完整规则配置，不包含当前轨迹数据无法表达的C类。

## 当前已完成

- Python项目和`uv`依赖定义；
- AV2运动预测数据与路径配置规范；
- 场景、参与者、地图、技能和过滤报告数据结构；
- 二维全局/局部坐标转换；
- 场景清单读写和跨split泄漏检查；
- 经用户确认的34类正式技能与5类候选规则YAML、AV2可行性矩阵和共享算子映射；
- 支持技能子集指纹和断点续跑的正式候选扫描；
- 旧30类与新增9类规则扫描结果的CSV/checkpoint验证和原子合并；
- 39规则扫描共生成97,664条种子候选记录，覆盖18,985个唯一场景；命中的34类全部进入正式技能库，正式技能种子覆盖为34/34；
- 5类零命中规则保留为候选，不进入当前正式训练、种子筛选或批量生成范围；
- 恰好5,000个最终种子场景及其33,914条完整正式技能标签；
- 内部验证、最终Validation零重叠及确定性、schema、checkpoint审计；
- 可在无显示器环境运行的BEV绘图器；
- 合成场景BEV冒烟脚本；
- 官方AV2测试场景的下载、读取和BEV验证脚本；
- 基于固定随机种子的AV2场景子集下载脚本；
- 前期单元测试、环境文档、数据文档、风险和参考资料清单。

当前已完成22,000个Train场景和5,000个Validation场景的确定性下载及全量可读性验证，并生成20,000/2,000正式划分及500/100开发子集清单。

## 环境

推荐在WSL2 Ubuntu中使用`uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd "/mnt/d/同济大学/Course/032 大三下/大数据智能分析"
uv python install 3.10
uv python pin 3.10
uv sync
uv run python -m pytest -q
```

`uv sync`会自动创建`.venv`。AV2接口属于项目正式依赖，pytest位于默认启用的`dev`依赖组，因此日常直接使用`uv run`即可，无需附加extra或手工激活环境。详细说明见[环境准备](docs/environment.md)。

## 无训练验证

运行全部准备阶段测试：

```bash
uv run python -m pytest -q
```

生成一个合成场景BEV检查图：

```bash
uv run python -m scripts.visualization.render_synthetic_bev
```

输出文件位于`outputs/synthetic_bev.png`，该目录不会提交Git。

验证官方AV2小型测试场景：

```bash
uv run python -m scripts.data.download_av2_test_sample
uv run python -m scripts.visualization.render_av2_sample \
  data/sample/av2/0a1e6f0a-1817-4a98-b02e-db8c9327d151/scenario_0a1e6f0a-1817-4a98-b02e-db8c9327d151.parquet
```

样例来自AV2 API官方仓库，总大小约220 KB，保存在被Git忽略的`data/`目录中。

真实数据采用确定性子集，不下载完整58GB数据集。Train准备流程为：

```bash
uv run python -m scripts.data.download_av2_subset \
  --s5cmd /mnt/c/Users/123456/.local/bin/s5cmd.exe \
  --split train --count 22000 --seed 2026 \
  --manifest manifests/acquisition/formal_train_pool.csv \
  --execute

uv run python -m scripts.data.split_av2_train_pool
```

第二条命令不下载或复制数据，只从正式Train清单固定划分20,000个训练、2,000个内部验证，并派生500/100开发子集。

完整下载命令见[AV2 数据下载命令](docs/data/download-commands.md)，容量和数据格式说明见[Argoverse 2数据准备](docs/data/argoverse2.md)。

## 目录

```text
configs/                  数据、路径和技能规则配置
docs/data/                AV2数据与场景清单说明
docs/goals/               按阶段编号的Goal和完整长期计划
docs/references/          论文、官方文档和开源代码清单
docs/skills/              技能候选、可行性矩阵和最终分类
skilldrive/data/          坐标、清单和单场景AV2适配器
skilldrive/schemas/       公共数据结构
skilldrive/skills/        技能YAML加载与校验
skilldrive/visualization/ BEV绘图
manifests/acquisition/    AV2下载池和后续扩容清单
manifests/splits/         正式训练、内部验证和最终Validation清单
manifests/development/    固定的500/100开发子集
manifests/seeds/          最终确定性种子清单
scripts/data/             AV2下载、划分和完整性验证命令
scripts/seed_detection/   候选扫描和正式种子筛选命令
scripts/visualization/    AV2、合成BEV和候选审核渲染命令
tests/unit/               按模块职责组织的单元测试
tests/workflows/          数据准备和种子检测流程测试
```

## 文档入口

- [01 前期准备Goal（已完成）](docs/goals/01_PREPARATION_GOAL.md)
- [02 技能体系设计Goal（已完成，当前34正式+5候选）](docs/goals/02_SKILL_LIBRARY_DESIGN_GOAL.md)
- [03 规则执行与候选种子检测Goal（已完成）](docs/goals/03_SKILL_SEED_DETECTION_GOAL.md)
- [最终34类正式技能与5类候选规则](docs/skills/skill-taxonomy.md)
- [39规则扩展扫描与34+5定稿记录](docs/seed-detection-expanded-review.md)
- [技能候选与决策记录](docs/skills/skill-candidates.md)
- [AV2技能可行性矩阵](docs/skills/av2-feasibility-matrix.md)
- [完整项目计划](docs/goals/FULL_PROJECT_PLAN.md)
- [环境准备](docs/environment.md)
- [Argoverse 2数据准备](docs/data/argoverse2.md)
- [场景清单与防泄漏](docs/data/manifest-format.md)
- [参考资料](docs/references/reading-list.md)
- [风险与待确认事项](docs/risks.md)
- [前期准备状态](docs/preparation-status.md)

## 范围限制

- 不提交原始AV2数据、完整权重或大规模生成结果；
- 不硬编码本地绝对路径；
- 未经用户明确确认，不进入模型训练阶段；
- 当前结果证明39个规则的扫描、34/34类正式技能种子覆盖和5,000场景筛选闭环可复现，不代表世界模型或下游性能已经得到验证。
