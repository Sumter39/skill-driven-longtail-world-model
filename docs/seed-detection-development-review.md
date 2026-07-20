# SkillDrive 开发集规则审核报告

## 1. 当前结论

用户已确认采用`1A + 2A + 3A + 4A + 5A`。对应实现、测试、固定500场景复扫和新版100张BEV审核均已完成。

最终开发结果相较首轮从3,749条降至1,461条，主要删除了同场景上的错误拓扑、同流路口关系、近重叠角色和弱运动标签。唯一场景数从494降至442，说明规则收紧主要减少噪声标签，没有耗尽开发场景基础。

用户已授权后续技术取舍采用推荐方案，因此已落实`6A`：`forced_lane_change_around_blockage`要求阻塞对象至少位于车辆前方2米。最终500场景审核已通过，下一步进入正式扫描准备。

## 2. 本轮已经落实的决定

### 2.1 方案1A：收紧基础结构

- 合流类只接受精确车道对的真实后继收敛，不再使用同车道跟车或任意lane ID回退；
- 分流类要求分流点位于车辆前方、横移量不少于0.25米，并且横移方向朝向被横跨车流；
- 路口类要求不同车流、30°–150°路径夹角、运动中的交叉车辆、至少2米当前间距和位于路口附近的可计算冲突点；
- 同步换道要求两辆车来自共享目标车道两侧，且中心距不少于2米；
- 慢车、停车、cut-out和阻塞类加入最低运动速度、有效闭合速度和至少2米的角色间距；
- 动态阻塞车必须与避让车位于同一路径关系；
- 数值上超出6秒风险时域的浮点TTC不再冒充目标风险观测。

### 2.2 方案2A：停车车辆三角色重入

`stopped_vehicle_reentry`现在固定使用：

1. `reentering_vehicle`；
2. `front_main_flow_vehicle`；
3. `rear_main_flow_vehicle`。

检测要求重入车在参考时刻仍处于连续停驻状态，并在同一主车流lane上找到运动中的前车和后车。

### 2.3 方案3A：允许多技能标签

同一场景可同时保留通用交叉冲突和具体左转冲突。候选记录按“场景＋技能＋参与者组合”区分，后续配额和实验统计仍可按场景去重。

### 2.4 方案4A：不放宽零命中A类

二轮共有8类A类零命中。新增的`jaywalking_pedestrian_crossing`零命中是因为首轮唯一候选属于同向平行误检，收紧后被正确删除。当前不降低阈值，等待固定规则在20,000场景上验证稀有性。

### 2.5 方案5A：区分种子风险与目标风险

候选使用：

- `seed_risk_metric`和`seed_risk_value`记录种子阶段实际可计算指标；
- `target_risk_definition_json`保留YAML定义的生成后目标指标；
- 当两者指标名称不同时，明确标记为代理风险。

最终1,461条候选中，739条使用代理指标，722条直接观测到目标指标。

## 3. 固定500场景复扫结果

| 指标 | 首轮基线 | 二轮结果 | 变化 |
|---|---:|---:|---:|
| 候选记录 | 3,749 | 1,461 | -2,288 |
| 唯一候选场景 | 494 | 442 | -52 |
| 命中技能 | 23/30 | 22/30 | -1 |
| A类候选 | 573 | 383 | -190 |
| B类候选 | 3,176 | 1,078 | -2,098 |
| 代理风险 | 1,718 | 739 | -979 |
| 目标指标观测 | 1,458 | 722 | -736 |
| 扫描时间 | 1,485.85秒 | 41.73秒 | -97.19% |

候选与2,000个内部验证场景、5,000个最终Validation场景的重叠均为0。

### 3.1 A类：已观察触发

| 技能 | 首轮 | 二轮 | 变化 |
|---|---:|---:|---:|
| `adjacent_vehicle_cut_in` | 0 | 0 | 0 |
| `bike_lane_vehicle_merge_conflict` | 1 | 1 | 0 |
| `crossing_path_conflict` | 3 | 3 | 0 |
| `crosswalk_pedestrian_crossing` | 1 | 1 | 0 |
| `cyclist_crossing` | 0 | 0 | 0 |
| `jaywalking_pedestrian_crossing` | 1 | 0 | -1 |
| `lead_hard_brake` | 0 | 0 | 0 |
| `lead_sudden_stop` | 101 | 79 | -22 |
| `narrow_gap_lane_change` | 0 | 0 | 0 |
| `ramp_merge_small_gap` | 27 | 2 | -25 |
| `rear_vehicle_rapid_approach` | 0 | 0 | 0 |
| `right_turn_vehicle_conflict` | 2 | 2 | 0 |
| `short_headway_following` | 48 | 48 | 0 |
| `slow_lead_blockage` | 386 | 244 | -142 |
| `turning_vehicle_crosswalk_conflict` | 0 | 0 | 0 |
| `unprotected_left_turn_conflict` | 3 | 3 | 0 |
| `wrong_way_vehicle` | 0 | 0 | 0 |

零命中的8类A类为：

`adjacent_vehicle_cut_in`、`cyclist_crossing`、`jaywalking_pedestrian_crossing`、`lead_hard_brake`、`narrow_gap_lane_change`、`rear_vehicle_rapid_approach`、`turning_vehicle_crosswalk_conflict`、`wrong_way_vehicle`。

### 3.2 B类：兼容基础种子

| 技能 | 首轮 | 二轮 | 变化 |
|---|---:|---:|---:|
| `construction_object_lane_blockage` | 19 | 19 | 0 |
| `cut_in_then_brake` | 162 | 162 | 0 |
| `cut_out_reveals_slow_vehicle` | 329 | 322 | -7 |
| `diverge_lane_crossing_conflict` | 341 | 59 | -282 |
| `forced_lane_change_around_blockage` | 304 | 203 | -101 |
| `intersection_blocking_vehicle` | 146 | 4 | -142 |
| `intersection_creep_conflict` | 374 | 5 | -369 |
| `lane_drop_merge_competition` | 260 | 11 | -249 |
| `merge_without_yield` | 257 | 6 | -251 |
| `roadside_pedestrian_emergence` | 78 | 78 | 0 |
| `simultaneous_lane_change_conflict` | 366 | 86 | -280 |
| `static_object_avoidance` | 91 | 91 | 0 |
| `stopped_vehicle_reentry` | 449 | 32 | -417 |

## 4. 数据和自动验证

- WSL项目`.venv`已恢复`dev+av2`依赖，`av2==0.3.6`可导入；
- 299项自动测试通过；
- Python编译检查通过；
- 真实AV2单场景端到端烟测通过；
- 500/500场景完成，checkpoint为501行且版本为2；
- CSV共1,461条，字段为schema 2，唯一键重复0；
- JSON字段解析失败0，非有限风险值0；
- 每条`target_risk_definition_json`均与对应技能YAML一致；
- 由未来TTC算子产生的目标TTC均限制在6秒风险时域内；
- 与内部验证和最终Validation重叠0；
- `git diff --check`通过；
- 未读取验证集调规则，未训练模型，未生成反事实轨迹。

## 5. 新版100张BEV审核

最终审核索引包含100条记录、72个唯一场景、22个有候选技能。目录中恰好保留100张有效PNG，`--restart`会自动清理旧索引外PNG。

人工查看全部10张联系表，并对同步换道、分流、受阻绕行和三角色重入的边界样本打开原图复核。

### 5.1 已确认改善

- 19条三类道路合流候选全部使用`two_source_lanes_share_successor`，未再出现同车道普通跟车；
- 路口阻塞4条、路口探入5条的路径夹角为69.91°–117.35°，当前间距为10.98–18.94米，均表现为真实交叉流；
- 同步换道86条的最小中心距为5.68米，未再出现近重叠车辆；
- cut-out的两段队列间距最小分别为2.016米和2.015米；
- 分流59条全部满足“横移方向朝向被横跨车流”；
- 三角色重入32条的前后角色顺序正确，前车最小轨迹距离不少于4.23米，后车不少于2.58米；
- 首轮唯一同向平行的jaywalking误检已经消失；
- BEV中的种子代理风险和生成后目标风险已分开显示。

### 5.2 合法但仍需说明的限制

- B类只证明基础拓扑、角色和可生成空间，不代表最终长尾动作已经自然发生；
- `roadside_pedestrian_emergence`仍是较弱反事实基础种子，无法从AV2轨迹证明遮挡原因；
- `unprotected_left_turn_conflict`只能证明左转与对向车流的几何冲突，无法证明动态信号相位；
- 500场景中的A类零命中不能直接证明正式Train中不存在该技能。

## 6. 方案6A落实结果

用户授权采用推荐方案后，检测器新增`minimum_blockage_distance_m: 2.0`：

- 阻塞对象速度不高于0.5 m/s；
- 避让车辆速度不低于1 m/s；
- 两者中心距不少于2米；
- 阻塞对象沿车辆行驶方向至少位于前方2米、至多35米；
- 动态阻塞车必须与避让车属于同一路径关系。

规则修改前有210条受阻绕行候选，其中15条前向距离不足2米。最终重扫得到203条，而不是简单减少15条，因为8个场景找到了其他合法参与者组合。

最终203条全部满足：

- 最小前向阻塞距离：2.044米；
- 最小车辆中心距：2.922米；
- 前向距离不足2米：0条。

最终100张BEV中抽到的受阻绕行样本均具有清晰前后关系，未再出现并排对象被标为“前方阻塞”。

## 7. 正式扫描结果

固定规则已完成20,000个正式Train场景扫描：

- 10进程总墙钟约12分47秒，最终平均26.78场景/秒；
- 原始候选池共55,322条记录，覆盖17,039个唯一场景和25/30类技能；
- 完成态checkpoint包含20,000个场景，再次运行相同命令可直接恢复而不重新读取场景；
- 确定性分层轮转最终选出5,000个唯一场景，并保留21,408条完整技能标签；
- 9,940条入选标签使用种子代理风险，11,468条直接观测目标风险；
- 候选池和最终种子清单与2,000个内部验证场景、5,000个最终Validation场景的重叠均为0；
- 候选顺序、反向输入选择一致性、CSV schema round trip和checkpoint城市补全均通过。

本Goal至此完成。后续工作进入条件轨迹生成模型阶段；当前仍未训练模型、未生成反事实轨迹，也未Push。
