# SkillDrive 规则执行与候选种子检测 Goal

## 1. 当前目标

本阶段针对`02_SKILL_LIBRARY_DESIGN_GOAL.md`中经用户确认的39个已实现规则，完成可复用的规则执行、参数采样和候选种子检测流程，并依据正式Train扫描结果定稿为34类正式技能与5类零命中候选规则。正式种子筛选只面向34类正式技能，先保留完整原始候选池，再确定性筛选5,000个可解释、可复现的唯一真实场景种子；选中场景的全部正式技能标签必须保留。

本阶段不重新设计技能语义或规则内容，只依据固定扫描证据和用户确认划分正式/候选状态；不训练条件CVAE，也不生成大规模反事实轨迹。

## 2. 开始条件

- 02 Goal已经完成；
- 39个待扫描规则以及正式/候选定稿原则经过用户确认；
- 每类具有完整YAML、AV2可行性说明和共享算子映射；
- 500/100开发清单和20,000/2,000正式清单保持固定；
- 当前自动测试全部通过。

## 3. 核心原则

- 检测器读取技能YAML，不在代码中复制另一套阈值；
- 优先实现共享几何、交互和风险算子，再组合成各技能规则；
- 技能语义由人工规则决定，数值阈值按照02 Goal记录的来源执行；
- 每个候选必须记录参与者角色、触发证据和风险指标；
- 同一真实场景可为多个技能提供种子，但必须分别记录技能和参与者组合；
- 不使用内部验证或最终Validation作为生成种子或调参依据。

## 4. 输出格式

候选种子清单至少包含：

```text
scenario_id
skill_id
initiator_track_id
responder_track_id
trigger_score
seed_risk_metric
seed_risk_value
target_risk_definition_json
source_path
evidence_json
sampled_parameters_json
```

输出必须确定性排序，相同配置和随机种子重复运行产生相同结果。

建议输出：

```text
outputs/seed_detection/development_candidate_pool.csv
outputs/seed_detection/formal_candidate_pool.csv
manifests/seeds/formal_candidates.csv
outputs/seed_detection/development_summary.json
outputs/seed_detection/formal_pool_summary.json
outputs/seed_detection/formal_summary.json
outputs/seed_detection/review/
```

开发与正式原始候选池和checkpoint属于可再生成的大体积产物，放在被Git忽略的`outputs/`目录；最终5,000场景种子清单放在`manifests/seeds/`中。

## 5. 执行步骤

### 阶段A：公共规则能力

- 建立候选种子数据结构和CSV读写；
- 实现经确认技能实际需要的最小几何、地图关系、相对运动和风险计算；
- 实现统一检测入口和拒绝原因统计；
- 实现读取YAML范围的确定性参数采样。

### 阶段B：规则映射与测试

- 将39个已实现规则分别映射到共享规则算子；
- 为共享算子编写正例、反例和边界测试；
- 为每个已实现规则至少建立一个代表性规则映射测试；
- 验证参数范围、参与者类型和缺失数据处理。

### 阶段C：500个开发场景扫描

- 扫描`manifests/development/development_train.csv`；
- 输出每类命中数、拒绝原因、城市、参与者和风险分布；
- 记录运行时间和峰值内存；
- 分层选择至少100个种子候选，并尽量覆盖当时有命中的规则，生成BEV审核材料。

### 阶段D：用户确认

- 汇报各技能命中情况、典型正确案例、误检和阈值问题；
- 对无候选、候选极少或误检严重的技能提供规则调整选项；
- 用户确认规则方向后才能扫描正式训练集。

### 阶段E：20,000个正式训练场景扫描

- 扫描`manifests/splits/formal_train.csv`；
- 先输出完整原始候选池和可续跑checkpoint；
- 根据39规则扫描结果确定正式与候选状态，5类零命中候选不进入正式种子池；
- 按正式`skill_id`、种子风险指标和风险值四分位进行确定性分层轮转，筛选恰好5,000个唯一真实场景；
- 同一场景一旦入选，保留该场景的全部正式技能匹配关系；
- 汇报34类正式技能、城市、参与者和风险区间覆盖，并单独报告5类候选的零命中事实；
- 检查与内部验证、最终Validation的场景ID零交集。

如果原始候选池不足5,000个唯一场景，不自动降低规则标准，也不自动下载更多数据；停止正式选择并报告证据。

### 阶段F：固化与交付

- 固化检测配置、随机种子和清单格式；
- 更新README、数据格式和阶段状态；
- 运行全部自动测试和`git diff --check`；
- 未经用户明确授权，不提交或推送Git。

## 6. 验收标准

- 39个已实现规则均有可执行规则映射和参数采样路径；
- 共享算子及39个规则的代表性映射测试通过；
- 500个开发场景完成扫描并输出分层BEV审核材料；
- 用户已经确认规则方向；
- 20,000个正式训练场景完成扫描；
- 34类正式技能获得恰好5,000个唯一候选种子，且保留这些场景的全部正式技能标签；若候选不足则形成明确证据说明并停止；
- 候选种子不包含内部验证和最终Validation场景；
- 输出可复现，全部自动测试通过；
- 没有训练模型或批量生成反事实轨迹。

## 7. Goal模式规则

1. 读取本文件、02 Goal和FULL计划，不重新设计技能目录。
2. 先实现共享规则能力，再组合技能，避免为每类复制近似代码。
3. 先测试，再扫描500个开发场景，用户确认后才扫描20,000个正式场景。
4. 用户已授权后续技术分歧默认采用推荐方案：记录证据和取舍后直接执行；只有扩大任务范围、删除数据、Commit、Push或其他需要新权限的操作才暂停确认。
5. 不读取内部验证或最终Validation调规则。
6. 不训练模型，不扩大下载规模。
7. 未经用户明确授权，不提交或推送Git。

## 8. 当前状态

截至2026-07-20，本Goal的阶段A至阶段F、39规则扩展扫描及34正式+5候选定稿均已完成：

- 当前保留39份完整YAML和可执行映射：正式技能34类，其中14类为`observed_trigger`、20类为`compatible_seed`；候选规则5类，均为本轮零命中的`observed_trigger`规则；
- 已完成共享几何、地图拓扑、风险计算、规则注册、参数采样、候选CSV、确定性续跑、候选池原子合并、泄漏审计和BEV审核工具；
- 30类历史基线的500场景开发审核、100张BEV和20,000场景正式扫描保持为可追溯证据：正式基线生成55,322条候选，覆盖17,039个唯一场景和25/30类；
- 根据用户“不限制总数”的决定，从原40类候选中恢复9类完成扫描；`wrong_way_cyclist`继续作为未实现历史候补，不属于当前5类候选规则；新增9类均具有独立YAML、可执行检测映射和参数采样路径；
- 新增9类先在200个正式Train场景上完成真实数据验证，再以10进程对20,000个正式Train场景执行独立增量扫描；
- 增量扫描使用技能子集指纹和独立checkpoint，支持中断续跑；最终生成42,342条候选，覆盖17,638个唯一场景，新增9类全部命中；
- `cyclist_vehicle_merge`的一侧辅助未来距离不可计算时保留为`null`并标记代理风险；至少一侧风险距离或TTC有效才保留，两侧均不可计算则拒绝；真实失败场景回归通过；
- 旧30类池与新增9类池经过清单指纹、技能集合、城市、场景顺序、CSV/checkpoint一致性和重复键校验后原子合并；
- 39规则历史合并扫描包含97,664条技能记录和18,985个唯一场景，其中34/39类获得真实触发或兼容基础种子，超过“至少30类覆盖”目标；
- 用户据此将命中的34类全部定为正式技能，实现34/34正式技能种子覆盖；零命中的`lead_hard_brake`、`rear_vehicle_rapid_approach`、`adjacent_vehicle_cut_in`、`narrow_gap_lane_change`和`wrong_way_vehicle`保留为候选规则，未为凑数降低阈值；
- 最终确定性筛选得到恰好5,000个唯一场景和33,914条正式技能标签，并保留入选场景的全部正式技能标签；
- 最终清单覆盖34/34类正式技能，其中新增稀有类保留`cyclist_vehicle_merge`2个、`group_pedestrian_crossing`3个和`motorcyclist_filtering_conflict`52个场景；
- 18,424条最终标签使用种子代理风险，15,490条直接观测目标风险；辅助量为`null`的案例不冒充完整风险真值；
- 正式候选池和最终清单均不包含2,000个内部验证或5,000个最终Validation场景；反向输入顺序确定性检查、CSV schema round trip和checkpoint匹配均通过；
- 旧30类结果和39规则扫描来源证据保存在`outputs/seed_detection/expanded_39/`；当前34类正式候选池、正式清单和验收汇总分别为`outputs/seed_detection/formal_candidate_pool.csv`、`manifests/seeds/formal_candidates.csv`和`outputs/seed_detection/formal_summary.json`；
- 完整30类开发审核证据见`docs/seed-detection-development-review.md`，39规则扩展扫描及34+5定稿证据见`docs/seed-detection-expanded-review.md`；
- 当前仍未训练模型、未生成反事实轨迹，也未Commit或Push。
