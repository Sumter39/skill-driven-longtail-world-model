# SkillDrive 39规则扩展扫描与34正式+5候选定稿结果

## 1. 结论

2026-07-20，技能库由30类历史基线扩展为39个已实现规则，其中A类19个、B类20个。新增9类均完成YAML、检测映射、参数采样、自动测试和20,000个正式Train场景增量扫描。该39规则扫描是后续34正式+5候选定稿的历史证据，不表示当前39类均为正式技能。

最终结果：

- 已实现规则：39个，其中34类正式技能、5类零命中候选规则；
- 正式技能：A类14个、B类20个，六族规模为4、4、6、6、7、7；
- 39规则历史扫描中有真实触发或兼容基础种子的规则：34/39类；
- 39规则合并扫描记录：97,664条、18,985个唯一场景；
- 34类正式技能种子覆盖：34/34类；
- 最终正式种子：5,000个唯一场景、33,914条正式技能标签；
- 与2,000个内部验证场景和5,000个最终Validation场景重叠均为0；
- 确定性反向输入检查、CSV schema往返和完整checkpoint匹配均通过；
- WSL下323项自动测试、关键源码编译和`git diff --check`均通过。

用户据此将命中的34类全部定为正式技能，并将5类零命中规则保留为候选。当前正式规则库满足“不少于30类”要求，且34/34类正式技能均获得真实或兼容种子覆盖。

## 2. 新增9类正式扫描结果

| 技能ID | 20,000场景候选数 | 最终5,000场景标签数 |
|---|---:|---:|
| `chain_braking` | 3,741 | 1,420 |
| `late_lane_change_before_diverge` | 6,941 | 2,276 |
| `zipper_merge_multi_vehicle` | 130 | 130 |
| `mutual_yield_deadlock` | 8,954 | 2,729 |
| `group_pedestrian_crossing` | 3 | 3 |
| `cyclist_vehicle_merge` | 2 | 2 |
| `abrupt_u_turn_conflict` | 13,594 | 3,850 |
| `multi_vehicle_gap_squeeze` | 8,925 | 3,098 |
| `motorcyclist_filtering_conflict` | 52 | 52 |

新增9类增量扫描共生成42,342条候选，覆盖17,638个唯一场景，9类全部命中。稀有类数量少但证据完整，因此保留全部有效场景，不通过降低阈值扩大数量。

## 3. 39规则扫描中零命中的5类

- `lead_hard_brake`；
- `rear_vehicle_rapid_approach`；
- `adjacent_vehicle_cut_in`；
- `narrow_gap_lane_change`；
- `wrong_way_vehicle`。

这些规则均保留完整YAML、检测映射、参数采样路径和自动测试，但已转为候选规则，不计入当前34类正式技能，也不进入正式训练、种子筛选或批量生成。当前阶段不把普通场景误标成长尾事件，也不为了得到39/39命中而降低标准。只有在新的固定数据池中获得合格种子、完成人工审核并经用户确认后，候选规则才能晋级。

## 4. `null`辅助证据的处理

真实扫描曾发现`cyclist_vehicle_merge`的一侧参与者缺少可共同计算的未来轨迹距离。处理规则为：

- 不使用无穷大、零或虚构数值填充，而是在辅助证据中写入`null`；
- 场景仍需满足真实横向并入、明确目标车道、前后机动车角色和当前间隙条件；
- 至少一侧未来风险距离或TTC有效才保留；两侧都不可计算则拒绝；
- 风险量不完整时，`seed_risk_metric`与目标指标不同，自动归为代理风险，不冒充完整风险真值。

最终5,000场景中有18,424条代理风险标签和15,490条直接目标风险观测。

## 5. 增量扫描与合并

新增9类使用独立输出和checkpoint：

```text
outputs/seed_detection/expanded_39/formal_incremental_9_candidate_pool.csv
outputs/seed_detection/expanded_39/formal_incremental_9.checkpoint.jsonl
outputs/seed_detection/expanded_39/formal_incremental_9_summary.json
```

扫描命令通过重复`--skill-id`只加载新增9类。checkpoint元数据记录选中的技能ID，并只对这些YAML计算技能指纹，因此无关技能文件变化不会破坏增量续跑。

旧30类池与新增9类池由`merge_candidate_pools.py`合并。当前正式合并额外排除5类经验证为零命中的候选规则；合并前验证：

- 两个checkpoint均完整覆盖20,000个正式场景；
- 正式清单指纹、场景顺序和城市一致；
- 两组技能ID互斥且合计39类；
- CSV记录与各自checkpoint逐场景一致；
- 被排除的5类在CSV和checkpoint中均为零命中；
- 合并后不存在重复候选键。

旧30类候选池、checkpoint和最终种子结果，以及新增9类的增量扫描产物，保存在`outputs/seed_detection/expanded_39/`中作为39规则扫描来源证据。当前34类正式结果为：

```text
outputs/seed_detection/formal_candidate_pool.csv
outputs/seed_detection/formal_pool_summary.checkpoint.jsonl
outputs/seed_detection/formal_pool_summary.json
manifests/seeds/formal_candidates.csv
outputs/seed_detection/formal_summary.json
```

5类候选规则清单保存在`configs/skills/candidate_catalog.yaml`。`expanded_39`目录名和其中的39规则checkpoint属于历史来源标识，不应重命名或覆盖。

## 6. 复现命令

只扫描新增9类时，使用：

```bash
uv run python -m scripts.seed_detection.detect_seeds \
  --manifest manifests/splits/formal_train.csv \
  --skill-id chain_braking \
  --skill-id late_lane_change_before_diverge \
  --skill-id zipper_merge_multi_vehicle \
  --skill-id mutual_yield_deadlock \
  --skill-id group_pedestrian_crossing \
  --skill-id cyclist_vehicle_merge \
  --skill-id abrupt_u_turn_conflict \
  --skill-id multi_vehicle_gap_squeeze \
  --skill-id motorcyclist_filtering_conflict \
  --workers 10 \
  --output-csv outputs/seed_detection/expanded_39/formal_incremental_9_candidate_pool.csv \
  --summary-json outputs/seed_detection/expanded_39/formal_incremental_9_summary.json \
  --checkpoint outputs/seed_detection/expanded_39/formal_incremental_9.checkpoint.jsonl \
  --confirm-formal-scan
```

相同命令默认续跑；只有检测代码、选中技能YAML或正式清单发生受控变化时才使用`--restart`。

从两组历史扫描来源重建当前34类正式候选池时，使用以下完整命令。5个`--exclude-skill-id`只允许排除已经验证为零命中的规则；命令会原子生成正式CSV、checkpoint和摘要：

```bash
uv run python -m scripts.seed_detection.merge_candidate_pools \
  --candidate-pool outputs/seed_detection/expanded_39/formal_candidate_pool_30_baseline.csv \
  --candidate-pool outputs/seed_detection/expanded_39/formal_incremental_9_candidate_pool.csv \
  --checkpoint outputs/seed_detection/expanded_39/formal_pool_summary_30_baseline.checkpoint.jsonl \
  --checkpoint outputs/seed_detection/expanded_39/formal_incremental_9.checkpoint.jsonl \
  --exclude-skill-id lead_hard_brake \
  --exclude-skill-id rear_vehicle_rapid_approach \
  --exclude-skill-id adjacent_vehicle_cut_in \
  --exclude-skill-id narrow_gap_lane_change \
  --exclude-skill-id wrong_way_vehicle \
  --output-csv outputs/seed_detection/formal_candidate_pool.csv \
  --output-checkpoint outputs/seed_detection/formal_pool_summary.checkpoint.jsonl \
  --output-summary-json outputs/seed_detection/formal_pool_summary.json
```

最终选择命令为：

```bash
uv run python -m scripts.seed_detection.select_formal_seeds
```

当前阶段没有训练模型、批量生成反事实轨迹或使用验证集调规则。
