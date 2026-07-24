# 06阶段下游轨迹预测与增强对比审查

## 1. 阶段结论

06阶段已经完成冻结数据合同下的恒速、LSTM和轻量矢量Transformer预测闭环，以及E0-E3四组、三个固定随机种子的正式训练和Final Validation一次性评估。

核心结论是：05阶段交付的1,512条过滤后轨迹可以直接用于下游训练，不需要先重训CVAE；质量过滤相对未过滤生成数据产生了明确收益，但当前规模和覆盖下，过滤后增强E3仍没有达到相对纯真实数据E0“真实长尾minFDE或Miss Rate改善至少5%”的预设目标。

- E3相对E0：真实长尾minADE改善`0.01558 m / 2.73%`，minFDE改善`0.01075 m / 0.93%`，Miss Rate增加`0.00322 / 2.31%`；
- E3相对E0：总体minFDE退化`0.01488 m / 0.39%`，Miss Rate增加`0.00527 / 0.84%`，均未超过2%容忍线；
- E3相对E2：真实长尾minFDE改善`0.09533 m / 7.65%`，按场景配对bootstrap的95%区间为`[-0.11230, -0.07870] m`，不跨0；
- 因此，当前证据支持“过滤能显著修复未过滤生成数据导致的长尾负迁移”，但不支持“1,512条过滤后数据已显著优于纯真实训练”。

阶段按负结果完成，不依据Final Validation继续调参、选择新checkpoint或修改长尾阈值。

## 2. 冻结合同与防泄漏

正式预测合同ID：

```text
a9dac35db64569f2df48895cd10c723c522b7e178aa0fb9cf20a532e2445e1ca
```

冻结内容包括：

- 真实训练缓存：29,382条样本，来自20,000个Formal Train场景；
- Internal Validation缓存：2,954条样本，来自2,000个场景；
- E1、E2、E3各增加1,512条样本，E0不增加生成样本；
- 轻量矢量Transformer、6模态输出、best-of-6 Smooth L1和模式分类损失；
- 每组1,000 optimizer step，固定种子`2026/2027/2028`；
- `batch_size=288`、2个持久DataLoader worker、prefetch 2、pinned memory和BF16 AMP；
- best checkpoint只由Internal Validation的`minFDE@6`选择；
- Final Validation在配置冻结和12个正式训练运行审计完成前未读取。

Formal Train、Internal Validation和Final Validation的场景ID零重叠。技能ID、过滤结论、生成来源和`proposal_mode`只用于数据审计与结果分层，没有进入预测模型输入。

## 3. 数据视图

| 组别 | 真实样本 | 新增样本 | 新增数据含义 |
| --- | ---: | ---: | --- |
| E0 | 29,382 | 0 | 纯真实训练基线 |
| E1 | 29,382 | 1,512 | 同来源槽位的无技能语义轻量随机扰动 |
| E2 | 29,382 | 1,512 | 从数值有效Prior候选中确定性选择，选择时不读取过滤结论 |
| E3 | 29,382 | 1,512 | 05阶段`balanced_accepted.jsonl`过滤后交付 |

E1有179条源未来不完整，共3,989个缺失点；这些位置使用Mask排除，没有伪造真值。E2和E3均为完整60步未来。E2不是故意挑选的失败集合，其实际accepted/rejected构成只在确定性选样之后关联，用于审计而不影响选择。

E3覆盖18个有合格生成结果的技能和704个源场景，其中：

- `learned_conditioned_prior`：638条；
- `rule_guided_prior_search`：874条。

第二类表示从无真实技能监督的CVAE Prior中用规则搜索合适轨迹，不能表述为模型已经学会相应技能条件。

## 4. 模型正确性与性能冻结

正确性证据：

- 16样本过拟合损失从`2.387`下降到`0.428`，末值/初值为`0.179`；
- 6模态输出、概率、Mask、恒速边界、best-of-6损失和checkpoint恢复测试通过；
- Internal Validation恒速基线：minADE `3.5557`、minFDE `9.1435`、Miss Rate `0.6747`；
- LSTM E0：minADE `1.4943`、minFDE `3.5319`、Miss Rate `0.5074`；
- Transformer E0开发结果：minADE `1.2875`、minFDE `2.9139`、Miss Rate `0.4766`，明显优于恒速后才进入正式对比。

训练性能在固定负载下比较batch size、worker、预取、持久worker、pinned memory和BF16。冻结配置的稳定吞吐中位数约`2,642 samples/s`，重复范围`2,619-2,852 samples/s`，峰值显存约`2.49 GiB`。batch 320/352没有稳定缩短端到端时间，352还出现`0.29 s`单步尖峰，因此没有保留更大的batch。正式单种子1,000步训练约129-143秒。

## 5. 正式训练审计

12个Transformer正式运行均存在`summary.json`、`latest.pt`和`best.pt`。`latest`均为1,000步；`best`来自冻结的100步验证节点，指纹与summary完全一致。

| 组别 | 2026 best step/minFDE | 2027 best step/minFDE | 2028 best step/minFDE |
| --- | ---: | ---: | ---: |
| E0 | 1000 / 2.9139 | 1000 / 2.8700 | 800 / 2.9348 |
| E1 | 900 / 2.8985 | 1000 / 2.8240 | 1000 / 2.7465 |
| E2 | 1000 / 2.9585 | 1000 / 2.9515 | 900 / 2.9028 |
| E3 | 900 / 2.9739 | 1000 / 2.9293 | 800 / 2.8936 |

表中数值只用于Internal Validation checkpoint选择，不是最终结论。

## 6. Final Validation构造

冻结后才构建Final Validation缓存：

- 5,000个官方Validation场景全部读取成功；
- 5,000条官方总体基准样本；
- 2,279条真实`observed_trigger`长尾样本，来自1,839个场景；
- 额外保存3,463条冻结检测原始记录，用于恢复每条长尾样本的风险指标；
- 79个张量分片，共7,279条评估样本；
- 14个模型/基线结果均包含完全相同的7,279个唯一sample ID。

34类正式技能中只有14类具有`observed_trigger`定义；20类`compatible_seed`只表示可生成上下文，不能伪装为真实发生事件。Final Validation中最终只有9类出现可用真实长尾样本。逐技能统计仍保留全部34类，零样本项明确记录为0。

## 7. Final Validation主结果

下表为三个固定种子的均值和样本标准差。Miss Rate使用6秒终点2米阈值。

### 7.1 总体5,000样本

| 组别 | minADE@6 | minFDE@6 | Miss Rate@6 |
| --- | ---: | ---: | ---: |
| E0 | 1.6415 ± 0.0349 | 3.7898 ± 0.0593 | 0.62647 ± 0.01200 |
| E1 | 1.6053 ± 0.0370 | 3.6817 ± 0.0854 | 0.62240 ± 0.00329 |
| E2 | 1.6363 ± 0.0128 | 3.7864 ± 0.0369 | 0.62687 ± 0.01373 |
| E3 | 1.6626 ± 0.0219 | 3.8047 ± 0.0391 | 0.63173 ± 0.01625 |

辅助基线：恒速为`4.7036 / 12.1972 / 0.8610`；单种子LSTM E0为`1.9125 / 4.5652 / 0.6674`。Transformer总体明显优于两种基础基线。

### 7.2 真实长尾2,279样本

| 组别 | minADE@6 | minFDE@6 | Miss Rate@6 |
| --- | ---: | ---: | ---: |
| E0 | 0.5712 ± 0.0335 | 1.1607 ± 0.1239 | 0.13910 ± 0.00534 |
| E1 | 0.5578 ± 0.0233 | 1.1480 ± 0.1268 | 0.14056 ± 0.00700 |
| E2 | 0.5926 ± 0.0522 | 1.2453 ± 0.1556 | 0.14319 ± 0.00579 |
| E3 | 0.5556 ± 0.0304 | 1.1500 ± 0.1017 | 0.14231 ± 0.00465 |

辅助基线：恒速为`1.2082 / 2.9509 / 0.2861`；单种子LSTM E0为`0.5367 / 1.1205 / 0.1334`。长尾切片集中于较规则、低速或跟驰类目标，因此它比官方总体切片绝对误差更低；不能据此认为长尾任务更简单或直接比较两个切片的绝对难度。

## 8. 配对变化与不确定性

表中“样本变化”是所有长尾样本的直接平均；95%区间先在每个场景内合并重复技能标签，再按`scenario_id`配对bootstrap 2,000次，因此场景加权均值与样本加权均值会有轻微差异。

| 比较 | 视图 | minFDE样本变化 | 相对变化 | 场景配对均值与95% CI |
| --- | --- | ---: | ---: | ---: |
| E1-E0 | 总体 | -0.10808 m | -2.85% | -0.10808 `[-0.13886, -0.07856]` |
| E1-E0 | 长尾 | -0.01273 m | -1.10% | -0.00926 `[-0.02803, 0.00862]` |
| E2-E0 | 总体 | -0.00340 m | -0.09% | -0.00340 `[-0.03218, 0.02642]` |
| E2-E0 | 长尾 | +0.08458 m | +7.29% | +0.08785 `[0.07040, 0.10529]` |
| E3-E0 | 总体 | +0.01488 m | +0.39% | +0.01488 `[-0.01900, 0.04576]` |
| E3-E0 | 长尾 | -0.01075 m | -0.93% | -0.00733 `[-0.02753, 0.01256]` |
| E3-E2 | 长尾 | -0.09533 m | -7.65% | -0.09518 `[-0.11230, -0.07870]` |

解释：

- E1在总体上改善明显，但长尾minFDE区间跨0、Miss Rate略差，说明简单随机扰动主要带来一般正则化，不能替代技能生成；
- E2在长尾上相对E0显著退化，证明数值可读取的CVAE轨迹不等于可用于训练的高质量轨迹；
- E3相对E2的长尾改善显著，支持地图、运动学、碰撞、风险和技能过滤的必要性；
- E3相对E0的minADE有小幅稳定改善，但minFDE区间跨0且Miss Rate略差，当前数据量不足以证明过滤后增强带来可靠长尾终点收益。

## 9. 风险与技能覆盖

真实长尾按各技能冻结风险目标区间和方向归一化后分为低、中、高三层，共`1,254 / 496 / 529`条；本轮3,463条检测记录的`seed_risk_metric`均与目标风险指标一致，没有代理指标样本。完整分层数值位于`manifests/prediction/final_evaluation_v1.json`。

E3训练覆盖与真实评估覆盖并不等价：

- `slow_lead_blockage`：E3训练300条，Final长尾1,675条，E3相对E0 minFDE改善约0.38%；
- `lead_sudden_stop`：E3训练210条，Final长尾480条，minFDE退化约2.00%；
- `short_headway_following`：E3训练121条，Final长尾61条，minFDE退化约1.39%；
- `crossing_path_conflict`：E3训练1条，Final长尾22条，minFDE改善约7.39%，但训练和评估样本都偏少；
- 其余多个技能只有1-8条Final样本，不能形成可靠单类结论；
- 9个有真实评估样本的技能中，有些没有E3 accepted训练样本；18个E3训练技能中，多数`compatible_seed`技能没有真实事件评估标签。

因此不能把当前总体变化解释为34类技能均已获得提升，也不能根据少量单类正结果追加事后结论。

## 10. 失败归因与后续决策

主成功门槛未达到的主要原因有：

1. E3仅1,512条，占29,382条真实样本约5%，增强信号较弱；
2. accepted只覆盖18类，且技能分布高度不平衡，每类1-300条；
3. 真实Final长尾由observed-trigger定义，和大量compatible-seed生成技能的语义覆盖不重合；
4. 05阶段508,640条有效Prior候选中只有1,560条通过过滤，说明CVAE地图和运动学质量边界仍然明显；
5. 当前模型是单目标、开放环、60步轨迹预测，不是联合多智能体或闭环规划世界模型；
6. 149条代表案例完成的是自动证据审查，不是独立人工语义复核。

下一步若继续研究，应另立生成器升级Goal，优先提高CVAE地图/运动学有效率、补齐零接受技能、扩大过滤后覆盖并改善observed-trigger与生成技能的语义对齐。不能通过放宽过滤阈值、复制样本或在Final Validation上继续调参来获得正结论。

## 11. 复现与产物

轻量、可提交证据：

```text
configs/prediction/formal_v1.json
manifests/prediction/input_audit_v1.json
manifests/prediction/augmentation_bundle_v1.json
manifests/prediction/formal_contract_v1.json
manifests/prediction/formal_run_audit_v1.json
manifests/prediction/final_validation_labels_v1.json
manifests/prediction/final_evaluation_v1.json
manifests/prediction/final_evaluation_audit_v1.json
```

被Git忽略的运行产物：

```text
/home/sumter/.cache/skilldrive/cvae_baseline/final_validation
/home/sumter/skilldrive-runtime-06/prediction/formal_v1
/home/sumter/skilldrive-runtime-06/prediction/final_labels_v1
/home/sumter/skilldrive-runtime-06/prediction/final_evaluation_v1
```

正式训练、Final标签扫描、评估与审计入口分别为：

```bash
PYTHONPATH=. uv run python -m scripts.prediction.train_suite ...
PYTHONPATH=. uv run python -m scripts.prediction.audit_formal_runs ...
PYTHONPATH=. uv run python -m scripts.prediction.scan_final_labels ...
PYTHONPATH=. uv run python -m scripts.prediction.evaluate_final ...
PYTHONPATH=. uv run python -m scripts.prediction.audit_final_evaluation ...
```

大型缓存、checkpoint和逐样本预测不进入Git。阶段关闭前仍以全量pytest、源码编译和`git diff --check`作为最终工程门禁。
