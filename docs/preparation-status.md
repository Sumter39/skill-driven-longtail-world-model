# 前期准备状态

更新时间：2026-07-19。

本文件记录01前期准备阶段的验收快照。后续02技能体系设计和03候选种子检测均已完成；当前34类正式技能、5类候选规则及39规则扫描状态见`goals/02_SKILL_LIBRARY_DESIGN_GOAL.md`、`goals/03_SKILL_SEED_DETECTION_GOAL.md`、`skills/skill-taxonomy.md`和`seed-detection-expanded-review.md`。

## 已完成

- 建立`pyproject.toml`、`.python-version`和`uv.lock`。
- 明确WSL2内使用`uv`管理Python 3.10和`.venv`。
- 建立AV2数据、路径和场景清单规范。
- 实现基于`s5cmd`列表和固定随机种子的AV2子集下载脚本。
- 实现从22,000个正式Train池派生20,000/2,000正式划分和500/100开发清单的脚本，不重复下载数据。
- 完成22,000个正式Train场景下载，并通过Parquet和地图JSON全量可读性验证。
- 完成5,000个官方Validation场景下载，并通过Parquet和地图JSON全量可读性验证。
- 生成20,000个正式训练、2,000个内部验证及500/100开发子集清单。
- 定义`Scenario`、`AgentTrack`、`MapPolyline`、`SkillSpec`和`FilterReport`。
- 实现全局/局部二维坐标转换与角度归一化。
- 实现场景清单CSV读写和跨split泄漏检查。
- 建立30类技能目录，完成5类核心技能YAML规范。
- 实现单场景AV2适配器，并通过官方测试场景验证。
- 实现无显示器BEV绘图器和合成场景冒烟脚本。
- 完成环境、AV2数据、清单格式、风险和参考资料文档。
- 更新README，使其准确反映“只完成前期准备、不训练”的状态。

## 已验证

- WSL uv环境：uv 0.11.29、Python 3.10.20、AV2 0.3.6可用。
- WSL uv环境运行全部测试：31项通过。
- Windows现有环境运行测试：24项通过，1项AV2 API测试因Windows未安装AV2而按设计跳过。
- Python源码通过`compileall`检查。
- `pyproject.toml`和`uv.lock`能够被标准工具读取。
- 技能目录包含30个唯一ID，其中5个标记为可执行。
- 坐标往返误差测试精度达到`1e-10`量级，优于`1e-4`验收线。
- 合成BEV已生成到`outputs/synthetic_bev.png`并完成视觉检查。
- 官方AV2测试场景已读取，包含58个参与者和213条地图折线；局部坐标BEV已生成到`outputs/av2_sample_bev.png`并完成视觉检查。
- 未创建`models/`、`training/`或`checkpoints/`目录。
- 未执行任何模型训练、批量生成或正式实验。
- 最终5,000个官方Validation场景已下载并验证，仅保留用于最终评估。

## 环境与样例验证结论

- `uv sync`已完成并产生`uv.lock`和Linux `.venv`。
- AV2 API所需的数据类字段和读取入口已通过自动测试。
- 官方小型测试场景的parquet和地图JSON能够加载。
- 历史/未来`observed`掩码能够保留。
- 地图边界、计算中心线和参与者轨迹在局部BEV中对齐。
- 样例只用于接口验证，不纳入未来训练或评价。

## 进入训练阶段前的门槛

- 后续GPU版PyTorch快速检查通过；
- 用户明确授权进入训练阶段。
