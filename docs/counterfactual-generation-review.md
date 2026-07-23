# 05阶段正式反事实生成审查

## 结论

2026-07-23，正式数值生成与质量过滤（阶段A-F）完成。正式合同只使用Formal Train，未打开Internal Validation或Final Validation清单，也未运行BEV绘图。

| 项目 | 结果 |
| --- | ---: |
| Formal Train场景 | 5,000 |
| 正式技能任务 | 33,914 |
| 每任务候选预算 | 16 |
| 候选预算总数 | 542,624 |
| 有效Prior候选并进入过滤 | 508,640 |
| Prior上下文无效候选 | 33,984 |
| 接受候选 | 1,560 |
| 拒绝候选 | 507,080 |
| 正式墙钟 | 4,994.3 s（约83.2 min） |
| BEV正式计时 | 未运行 |

无效Prior候选的原因是目标参与者在历史结束帧（frame 49）不可见。它们只保留任务、候选索引、随机种子、原因码和哈希，不伪造轨迹、不读取目标真实未来补齐输入。

## 合同与执行

- Formal plan ID：`6b2da617bcf0694b87ea055285f971b58d660ae4591f49d039de1d51de99baf3`
- task plan SHA-256：`daff09d2b418039f245949eba9cf8a8d261b630123bfa1c77d93eceaf6f17909`（以运行目录的`formal_task_plan.summary.json`为准）
- CUDA Prior生成，task batch 64，CPU过滤进程8，map batch 32，`resume_mode=auto`
- 原子raw、过滤提交和每任务状态均绑定源、配置、checkpoint和合同哈希
- 运行目录：WSL ext4下的`/home/sumter/skilldrive-runtime-05/outputs/generation/counterfactual_v1/formal/formal_v1/6b2da617bcf0694b87ea055285f971b58d660ae4591f49d039de1d51de99baf3`
- 项目归档：`outputs/generation/formal_v1_6b2da617bcf0694b87ea055285f971b58d660ae4591f49d039de1d51de99baf3.tar`
- 归档SHA-256：`3533bfa2212ba5fca48580388a16c02ed93eed088e98303d97835f72af289aa5`

校验结果：33,914个任务全部为`accepted`或`rejected`，失败任务数为0；34类技能均有任务状态。`group_pedestrian_crossing`的3个任务全部因Prior上下文无效而拒绝，因此没有过滤目录，这是明确负结果而非漏项。

## 过滤漏斗

拒绝候选的首个失败阶段如下。阶段采用严格短路，后续阶段不对已拒绝候选重复计算。

| 首个失败阶段 | 候选数 |
| --- | ---: |
| `map` | 194,162 |
| `kinematics` | 191,792 |
| `collision` | 58,958 |
| `target_risk` | 44,959 |
| `skill_trigger` | 14,142 |
| `parameter_realization` | 2,215 |
| `diversity` | 852 |

## 逐技能结果

`valid`是进入过滤的候选数；`invalid`是因Prior上下文无效保留的候选数。接受数是候选数，不是任务数。

| 技能 | 任务 | valid | accepted | rejected | invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| abrupt_u_turn_conflict | 3,850 | 57,744 | 0 | 57,744 | 3,856 |
| bike_lane_vehicle_merge_conflict | 14 | 208 | 0 | 208 | 16 |
| chain_braking | 1,420 | 18,320 | 1 | 18,319 | 4,400 |
| construction_object_lane_blockage | 525 | 7,792 | 20 | 7,772 | 608 |
| crossing_path_conflict | 120 | 1,920 | 1 | 1,919 | 0 |
| crosswalk_pedestrian_crossing | 75 | 1,056 | 3 | 1,053 | 144 |
| cut_in_then_brake | 2,201 | 35,184 | 0 | 35,184 | 32 |
| cut_out_reveals_slow_vehicle | 3,683 | 55,616 | 14 | 55,602 | 3,312 |
| cyclist_crossing | 7 | 112 | 0 | 112 | 0 |
| cyclist_vehicle_merge | 2 | 32 | 0 | 32 | 0 |
| diverge_lane_crossing_conflict | 860 | 13,760 | 0 | 13,760 | 0 |
| forced_lane_change_around_blockage | 2,721 | 42,608 | 183 | 42,425 | 928 |
| group_pedestrian_crossing | 3 | 0 | 0 | 0 | 48 |
| intersection_blocking_vehicle | 41 | 656 | 0 | 656 | 0 |
| intersection_creep_conflict | 144 | 2,304 | 5 | 2,299 | 0 |
| jaywalking_pedestrian_crossing | 16 | 144 | 3 | 141 | 112 |
| lane_drop_merge_competition | 355 | 5,168 | 28 | 5,140 | 512 |
| late_lane_change_before_diverge | 2,276 | 34,144 | 0 | 34,144 | 2,272 |
| lead_sudden_stop | 1,077 | 17,232 | 210 | 17,022 | 0 |
| merge_without_yield | 468 | 6,944 | 39 | 6,905 | 544 |
| motorcyclist_filtering_conflict | 52 | 832 | 3 | 413 | 416 |
| multi_vehicle_gap_squeeze | 3,098 | 45,024 | 335 | 44,689 | 4,544 |
| mutual_yield_deadlock | 2,729 | 43,664 | 0 | 40,976 | 2,688 |
| ramp_merge_small_gap | 28 | 448 | 0 | 432 | 16 |
| right_turn_vehicle_conflict | 20 | 320 | 0 | 320 | 0 |
| roadside_pedestrian_emergence | 1,157 | 12,224 | 0 | 12,224 | 6,288 |
| short_headway_following | 875 | 12,368 | 121 | 12,247 | 1,632 |
| simultaneous_lane_change_conflict | 1,022 | 16,352 | 0 | 16,304 | 48 |
| slow_lead_blockage | 2,947 | 47,152 | 313 | 46,839 | 0 |
| static_object_avoidance | 1,264 | 20,224 | 258 | 18,590 | 1,376 |
| stopped_vehicle_reentry | 603 | 9,648 | 5 | 9,643 | 0 |
| turning_vehicle_crosswalk_conflict | 35 | 560 | 0 | 496 | 64 |
| unprotected_left_turn_conflict | 96 | 1,536 | 0 | 1,536 | 0 |
| zipper_merge_multi_vehicle | 130 | 1,952 | 18 | 1,934 | 128 |

对每个技能，`valid = accepted + rejected`，且`valid + invalid = 任务数 × 16`。该表用于定位能力边界，不用于把低接受率技能删除出正式技能库。

## 性能与恢复

性能冻结同时考虑绝对节省、相对变化、重复波动和额外复杂度。正式运行在ext4目录完成，避免在`/mnt/d`上产生数万个小文件写入；正式计算结束后才顺序归档到项目目录。运行中断烟测验证了raw分片和无效候选sidecar可以由同一命令恢复。

后续代码改进只影响未来恢复体验：过滤ETA按阶段实际候选数计算，全部任务已持久化时恢复入口先短路，不会增加正式运行的计算路径。相关小测试、状态测试和原始分片测试共43项通过，目标文件通过`compileall`；项目全量870项测试全部通过。

## 未完成与限制

1. 阶段G的逐技能代表性BEV、人工审核和最多300条/技能的确定性平衡清单尚未生成。
2. 1,560条接受候选只表示通过当前过滤合同，不表示34类技能都已被模型可靠控制；必须结合逐技能对照和失败边界分析。
3. 当前合同是单目标、开放环overlay，背景参与者不会对生成目标重新规划；不能宣传为联合闭环世界模型。
4. 下游预测器训练、E0-E3对比和最终Validation评估尚未开始。
