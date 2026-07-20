# SkillDrive 最终技能体系

## 1. 决策状态

2026-07-19，用户首先确认候选文档中的“平衡方案A”，形成30类规则基线。2026-07-20，用户进一步确认恢复候选池且不限制总数，将原候补中的9类加入39规则扫描。扫描后用户依据真实或兼容种子覆盖定稿：当前体系包含34类正式技能与5类零命中候选规则；正式技能中A类14个、B类20个，正式六族规模为4、4、6、6、7、7。正式与候选合计保留39份完整规则配置，其中A类19个、B类20个。

- A类：可从AV2直接观察和筛选自然或边界种子，并由轨迹模型生成和评价。
- B类：使用AV2兼容基础场景，主要通过反事实条件生成形成长尾行为，仍可自动评价。
- C类：当前数据无法可靠表达，不进入39个已实现规则，也不计入34类正式技能或5类候选规则。

34类正式技能与5类候选规则都有独立YAML和完整字段，不存在名称占位或`implemented: false`条目。候选规则保留实现与测试，但不进入当前正式训练、种子筛选或批量生成范围。

## 2. 正式34类

### 2.1 纵向车辆交互

| 技能ID | 中文名称 | 等级 |
|---|---|:---:|
| `lead_sudden_stop` | 前车突然停车 | A |
| `slow_lead_blockage` | 慢车阻塞 | A |
| `short_headway_following` | 短时距跟车 | A |
| `chain_braking` | 多车制动传播 | B |

### 2.2 换道与横向交互

| 技能ID | 中文名称 | 等级 |
|---|---|:---:|
| `cut_out_reveals_slow_vehicle` | 前车切出后暴露慢车 | B |
| `simultaneous_lane_change_conflict` | 双车同时换入同一车道 | B |
| `forced_lane_change_around_blockage` | 受阻绕行换道 | B |
| `late_lane_change_before_diverge` | 分流前晚换道 | B |

### 2.3 汇入、分流与车道拓扑冲突

| 技能ID | 中文名称 | 等级 |
|---|---|:---:|
| `ramp_merge_small_gap` | 小间隙汇入 | A |
| `lane_drop_merge_competition` | 车道收敛汇入竞争 | B |
| `merge_without_yield` | 汇入未让行 | B |
| `diverge_lane_crossing_conflict` | 分流横跨冲突 | B |
| `bike_lane_vehicle_merge_conflict` | 自行车道与机动车汇合冲突 | A |
| `zipper_merge_multi_vehicle` | 多车交替汇入 | B |

### 2.4 路口车辆交互

| 技能ID | 中文名称 | 等级 |
|---|---|:---:|
| `unprotected_left_turn_conflict` | 无保护左转冲突 | A |
| `right_turn_vehicle_conflict` | 右转车辆与交叉车流冲突 | A |
| `crossing_path_conflict` | 路口交叉路径冲突 | A |
| `intersection_creep_conflict` | 路口低速探入冲突 | B |
| `intersection_blocking_vehicle` | 车辆滞留路口冲突区 | B |
| `mutual_yield_deadlock` | 双方互让僵持 | B |

### 2.5 行人与骑行者交互

| 技能ID | 中文名称 | 等级 |
|---|---|:---:|
| `crosswalk_pedestrian_crossing` | 人行横道行人横穿 | A |
| `jaywalking_pedestrian_crossing` | 非人行横道行人横穿 | A |
| `roadside_pedestrian_emergence` | 路侧行人突然进入道路 | B |
| `cyclist_crossing` | 骑行者横穿 | A |
| `turning_vehicle_crosswalk_conflict` | 转向车辆与横道人行冲突 | A |
| `group_pedestrian_crossing` | 多人群体横穿 | A |
| `cyclist_vehicle_merge` | 骑行者并入机动车流 | A |

### 2.6 异常运动、粗粒度阻塞与组合长尾

| 技能ID | 中文名称 | 等级 |
|---|---|:---:|
| `stopped_vehicle_reentry` | 停车车辆重新进入车流 | B |
| `construction_object_lane_blockage` | construction类别占道阻塞 | B |
| `static_object_avoidance` | 粗粒度静态对象避让 | B |
| `cut_in_then_brake` | 切入后立即制动 | B |
| `abrupt_u_turn_conflict` | 突然掉头冲突 | B |
| `multi_vehicle_gap_squeeze` | 多车间隙夹逼 | B |
| `motorcyclist_filtering_conflict` | 摩托车狭缝穿行冲突 | B |

## 3. 候选规则5类

以下5类均具有完整YAML、检测映射、参数采样路径和自动测试，但在固定20,000个正式Train场景的39规则扫描中命中数为0，因此不计入当前正式技能库：

| 技能ID | 中文名称 | 原可行性 | 当前状态 |
|---|---|:---:|---|
| `lead_hard_brake` | 前车急减速 | A | 零命中候选 |
| `rear_vehicle_rapid_approach` | 后车快速逼近 | A | 零命中候选 |
| `adjacent_vehicle_cut_in` | 相邻车辆切入 | A | 零命中候选 |
| `narrow_gap_lane_change` | 窄间隙换道 | A | 零命中候选 |
| `wrong_way_vehicle` | 车辆持续逆向行驶 | A | 零命中候选 |

候选规则不会因为零命中而降低阈值，也不会使用普通场景冒充长尾事件。只有在新的固定Train数据池中获得合格种子、完成人工审核并经用户确认晋级后，才可加入正式训练、种子筛选和批量生成范围。`wrong_way_cyclist`仍是未实现历史候补，不属于这5类候选规则。

## 4. 共享能力

34类正式技能与5类候选规则不对应39个独立模型，而是组合以下共享能力：

| 共享能力 | 用途 |
|---|---|
| `LONGITUDINAL` | 加速、减速、停车、跟车和重新起步 |
| `LANE_CHANGE` | 横向换道、切入、绕行和回正 |
| `MERGE` | 汇入、分流和间隙接受 |
| `CONFLICT_POINT` | 计算和控制参与者到达冲突点的时序 |
| `YIELD_PRIORITY` | 表达让行、抢行和优先角色 |
| `VRU_CROSSING` | 行人或骑行者横穿和进入道路 |
| `BLOCKAGE_RESPONSE` | 对低速、停车或静态占道条件的响应 |
| `MULTI_AGENT` | 三名及以上参与者的联合时序控制 |

共享能力复用几何和生成机制；技能YAML保留不同触发条件、参与者角色、参数、风险目标、约束和期望行为。

## 5. 数据依据

500个Train开发场景的实测结果包括：

- 车辆轨迹20,967条，行人轨迹2,822条，骑行者轨迹230条；
- 403个场景含车辆与行人，134个含车辆与骑行者；
- 459个场景含人行横道，496个含路口车道；
- 206个场景含自行车道，499个含相邻车道结构；
- 486条construction轨迹中437条总位移不超过1米；
- 2,012条static轨迹中1,562条总位移不超过1米。

详细证据见`av2-feasibility-matrix.md`。

20,000个正式Train场景的39规则历史扫描结果与最终定稿为：

- 合并候选池包含97,664条技能记录和18,985个唯一场景；
- 39个已实现规则中34类获得真实触发或兼容基础种子，满足“至少30类有种子覆盖”的阶段目标；
- 新增9类全部命中，其中骑行者并流2个场景、群体行人横穿3个场景、摩托车穿行52个场景；
- 命中的34类全部进入正式技能库，实现34/34正式技能种子覆盖；零命中的5类转为候选规则，不降低阈值伪造真实命中。

## 6. 数据范围边界

最终体系不包含以下C类：

- 雨、雾、眩光和低照度；
- 相机模糊、损坏和曝光异常；
- 依赖精细对象外形的碎片、低矮障碍物和具体锥桶；
- 临时封道、临时改道和临时限速；
- 依赖动态灯色的闯红灯；
- 依赖车辆身份的应急车辆优先权；
- 需要视觉可见性证明的遮挡后显现。

`construction_object_lane_blockage`和`static_object_avoidance`只使用AV2粗粒度类别和保守安全缓冲，不声称数据能够识别具体障碍物。

## 7. 配置契约

每份技能YAML必须包含：

```text
skill_id
name_zh
family
definition
source
data_support
seed_requirements
trigger
detection
actors
parameters
generation_operators
constraints
risk_definition
expected_behavior
validation_metrics
known_limitations
output_labels
```

规则来源限定为课程示例、交通规则、安全指标、相关文献和Train数据模式。数值阈值必须注明`semantic`、`train_statistics`或`reference`来源。

## 8. 阶段结果与后续入口

`../goals/03_SKILL_SEED_DETECTION_GOAL.md`已经完成39规则执行、增量正式扫描、34+5定稿和最终种子重选：

1. 旧30类基线扫描得到55,322条候选并覆盖25类；
2. 新增9类增量扫描得到42,342条候选且9类全部命中；
3. 39规则合并扫描产生97,664条候选记录，其中命中的34类定为正式、零命中5类转为候选；
4. 从34类正式技能候选中确定性筛选5,000个唯一场景，最终清单保留33,914条正式技能标签并覆盖34/34正式技能，与内部验证和最终Validation零重叠。

下一阶段进入条件CVAE轨迹世界模型的数据读取、训练准备和小规模过拟合验证；当前仍未训练模型或生成反事实轨迹。
