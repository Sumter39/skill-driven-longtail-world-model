# SkillDrive

**Skill-Driven Long-Tail Scenario Generation for Autonomous Driving**

SkillDrive研究如何使用技能规则和学习式轨迹世界模型，从真实驾驶片段构造可控、连续并符合交通约束的长尾场景。

当前仓库已经完成**前期准备、39个规则实现、正式候选种子检测、条件CVAE基线、反事实生成与过滤，以及06下游轨迹预测正式对比**。05阶段从542,624条候选预算中接受1,560条并平衡交付1,512条，149条代表案例完成自动证据审查，但不宣称独立人工语义复核。06阶段已经冻结并审计E0-E3数据合同，完成四组、三个固定种子的正式训练，以及5,000个Final Validation场景上的一次性评估。结果显示过滤后E3相对未过滤E2的真实长尾minFDE改善`0.0953 m / 7.65%`，但相对纯真实E0只改善`0.0107 m / 0.93%`，没有达到预设5%目标；该负结果与能力边界完整保留。

## 技术路线

```text
Argoverse 2 Motion Forecasting
        ↓
矢量地图与多智能体历史轨迹
        ↓
技能规则条件
        ↓
条件CVAE轨迹世界模型（基线已训练）
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
- 条件CVAE的确定性v5张量缓存、地图上下文去重、场景编码、条件先验/后验、轨迹解码、训练、独立评估、原子checkpoint和中断续训链路；
- 500/100开发缓存与训练闭环，以及20,000个正式Train场景和2,000个Internal Validation场景的全流程缓存；
- 正式训练共50轮、2,200个optimizer step，冻结配置为`batch_size=672`、`num_workers=8`、持久worker和BF16 AMP；
- 2,000个基础内部验证样本上的Prior `minADE@6/minFDE@6`为`2.276843/5.306342 m`，954个真实技能样本上为`0.731041/1.475967 m`；
- 16个固定内部验证样例的Posterior、6条Prior轨迹和BEV诊断，图片与推理轨迹均记录SHA-256；
- 可在无显示器环境运行的BEV绘图器；
- 合成场景BEV冒烟脚本；
- 官方AV2测试场景的下载、读取和BEV验证脚本；
- 基于固定随机种子的AV2场景子集下载脚本；
- 前期单元测试、环境文档、数据文档、风险和参考资料清单。
- 下游恒速、LSTM和轻量矢量Transformer，6模态预测与统一minADE/minFDE/Miss Rate评估；
- E0真实、E1随机增强、E2未过滤生成、E3过滤后生成四组公平数据视图，各增强组精确1,512条；
- 12个正式Transformer训练运行及latest/best checkpoint审计，固定1,000步和`2026/2027/2028`三个种子；
- 5,000条Final Validation总体样本和2,279条真实长尾样本的逐技能、风险层与2,000次按场景配对bootstrap评估。

当前已完成22,000个Train场景和5,000个Validation场景的确定性下载及全量可读性验证，并生成20,000/2,000正式划分及500/100开发子集清单。

## 条件CVAE基线结果

v5缓存严格区分真实发生的`observed_trigger`与仅表示可生成性的`compatible_seed`；后者以及`sampled_parameters_json`均未作为真实技能监督。最终Validation在本阶段未被访问。

| 分区 | 场景 | 基础样本 | 真实技能样本 | 拒绝技能样本 | 缓存耗时 | 占用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development Train | 500 | 500 | 272 | 111 | 44.07 s | 63.19 MiB |
| Development Validation | 100 | 100 | 44 | 22 | 12.42 s | 11.78 MiB |
| Formal Train | 20,000 | 20,000 | 9,382 | 4,764 | 1,749.84 s | 2,405.66 MiB |
| Internal Validation | 2,000 | 2,000 | 954 | 493 | 298.98 s | 241.80 MiB |

固定active benchmark contract `fb0bca8...`、20步预热、200步计时并重复3次后，最终训练配置的吞吐为`1958.21/1892.71/1926.85 samples/s`，中位数`1926.85 samples/s`。正式训练墙钟为`808.67 s`，训练计时为`753.17 s`，峰值显存`6682.32 MiB`。基础样本的恒速ADE/FDE为`4.613611/11.943377 m`，学习式Prior的`minADE@6/minFDE@6`为`2.276843/5.306342 m`。训练摘要中的`final_validation_loss`字段表示最后一次Internal Validation损失，不代表访问了官方Final Validation split。这些结果证明基线训练与概率轨迹生成闭环成立，不代表34类技能已经全部实现可靠可控生成。

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

## 验证与复现

运行全部准备阶段测试：

```bash
uv run python -m pytest -q
```

06阶段的大型缓存、checkpoint和逐样本结果保存在WSL ext4并由Git忽略；轻量正式结果和审计位于`manifests/prediction/`。完整实验口径、数值和复现入口见[06阶段下游预测审查](docs/downstream-prediction-review.md)。

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

正式生成完成后的审查（只读取已完成运行目录，不重新生成候选）：

```bash
RUN_ROOT=/home/sumter/skilldrive-runtime-05/outputs/generation/counterfactual_v1/formal/formal_v1/6b2da617bcf0694b87ea055285f971b58d660ae4591f49d039de1d51de99baf3
PYTHONPATH=. uv run python -m scripts.generation.review_formal_output --run-root "$RUN_ROOT"
PYTHONPATH=. uv run python -m scripts.generation.select_formal_delivery --run-root "$RUN_ROOT"
PYTHONPATH=. uv run python -m scripts.generation.audit_formal_review \
  "$RUN_ROOT/review/formal_review_v1/summary.json"
# 人工填写 manual_review.csv 后，再固化至少100条人工结论：
PYTHONPATH=. uv run python -m scripts.generation.finalize_formal_review \
  "$RUN_ROOT/review/formal_review_v1/summary.json" \
  "$RUN_ROOT/review/formal_review_v1/manual_review.csv"
# 用户已授权自动证据审查时：
PYTHONPATH=. uv run python -m scripts.generation.run_automated_formal_review \
  "$RUN_ROOT/review/formal_review_v1/summary.json" --run-root "$RUN_ROOT"
PYTHONPATH=. uv run python -m scripts.generation.finalize_formal_review \
  "$RUN_ROOT/review/formal_review_v1/summary.json" \
  "$RUN_ROOT/review/formal_review_v1/automated_review.csv" \
  --output "$RUN_ROOT/review/formal_review_v1/automated_review_summary.json" \
  --review-method automated_evidence
```

审查图片、清单和人工审查模板均写入被Git忽略的运行目录。每条已审案例分别记录历史不变量、道路关系、运动连续性、技能角色、目标风险、参数实现度、背景交互和视觉异常8列，取值为`pass`、`fail`、`not_applicable`或`uncertain`；总体`review_status`使用`passed`、`failed`或`uncertain`。本轮经用户授权使用`codex-automated-evidence-v1`完成05 Goal的自动证据审查；`manual_review.csv`仍保留为后续独立人工复核模板，自动结果不得改称人工结论。

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
configs/                  数据、路径、技能规则和模型配置
docs/data/                AV2数据与场景清单说明
docs/goals/               按阶段编号的Goal和完整长期计划
docs/references/          论文、官方文档和开源代码清单
docs/skills/              技能候选、可行性矩阵和最终分类
skilldrive/data/          坐标、清单、AV2适配器和CVAE张量缓存
skilldrive/models/        条件CVAE模型
skilldrive/training/      损失、指标、训练与checkpoint
skilldrive/schemas/       公共数据结构
skilldrive/skills/        技能YAML加载与校验
skilldrive/visualization/ BEV绘图
manifests/acquisition/    AV2下载池和后续扩容清单
manifests/splits/         正式训练、内部验证和最终Validation清单
manifests/development/    固定的500/100开发子集
manifests/seeds/          最终确定性种子清单
scripts/data/             AV2下载、划分和完整性验证命令
scripts/modeling/         CVAE预处理、训练、评估、诊断和性能基准
scripts/seed_detection/   候选扫描和正式种子筛选命令
scripts/visualization/    AV2、合成BEV和候选审核渲染命令
scripts/generation/       反事实生成、正式结果审查和交付选择命令
skilldrive/prediction/    下游预测数据、模型、损失、指标和评估
scripts/prediction/       E0-E3预处理、训练、正式评估和审计入口
tests/unit/               按模块职责组织的单元测试
tests/workflows/          数据准备、种子检测和模型训练流程测试
```

## 文档入口

- [01 前期准备Goal（已完成）](docs/goals/01_PREPARATION_GOAL.md)
- [02 技能体系设计Goal（已完成，当前34正式+5候选）](docs/goals/02_SKILL_LIBRARY_DESIGN_GOAL.md)
- [03 规则执行与候选种子检测Goal（已完成）](docs/goals/03_SKILL_SEED_DETECTION_GOAL.md)
- [04 条件CVAE基线优化与正式训练Goal（已完成）](docs/goals/04_CONDITIONAL_CVAE_BASELINE_GOAL.md)
- [05 批量反事实生成与质量过滤Goal（已完成，自动审查不等同于独立人工复核）](docs/goals/05_COUNTERFACTUAL_GENERATION_FILTERING_GOAL.md)
- [06 下游轨迹预测、增强对比与性能冻结Goal（已完成）](docs/goals/06_DOWNSTREAM_TRAJECTORY_PREDICTION_GOAL.md)
- [06阶段下游轨迹预测与增强对比审查](docs/downstream-prediction-review.md)
- [05阶段正式生成审查](docs/counterfactual-generation-review.md)
- [条件CVAE基线阶段审查](docs/cvae-baseline-review.md)
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
- 当前正式数值生成、代表案例BEV自动审计、自动证据审查、确定性平衡选择和下游预测对比已完成；本轮自动审查的限制已记录；
- 当前结果证明生成质量过滤相对未过滤数据能减少真实长尾负迁移，但1,512条E3相对纯真实E0没有达到5%长尾改善目标，也不代表34类技能均已实现可靠控制或独立提升。
