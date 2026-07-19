# 前期参考资料清单

检索日期：2026-07-19。学术论文通过OpenAlex聚合的CrossRef/arXiv元数据检索，并按DOI去重；官方数据和代码链接以项目主页为准。

## 数据集与接口

1. **Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting**
   Wilson et al., 2023. DOI: [10.48550/arXiv.2301.00493](https://doi.org/10.48550/arXiv.2301.00493)
   用途：AV2数据规模、运动预测任务、地图与场景定义。
   代码：[argoverse/av2-api](https://github.com/argoverse/av2-api)，可直接复用数据读取与地图API。

## 轨迹生成与预测

2. **Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data**
   Salzmann et al., ECCV 2020. DOI: [10.1007/978-3-030-58523-5_40](https://doi.org/10.1007/978-3-030-58523-5_40)
   用途：条件潜变量、多智能体交互、动力学可行轨迹生成。
   代码：[StanfordASL/Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus)，只参考设计，不在准备阶段复现。

3. **VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation**
   Gao et al., CVPR 2020. DOI: [10.48550/arXiv.2005.04259](https://doi.org/10.48550/arXiv.2005.04259)
   用途：矢量车道和参与者折线编码，为后续轻量矢量Transformer提供结构依据。
   代码：检索结果未确认权威官方实现，避免直接依赖第三方复现。

4. **Auto-Encoding Variational Bayes**
   Kingma and Welling, 2013. DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
   用途：CVAE潜变量、重参数化和KL项的理论基础。
   代码：不需要复用外部代码。

## 长尾与安全关键场景生成

5. **AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles**
   Wang et al., CVPR 2021. DOI: [10.1109/CVPR46437.2021.00978](https://doi.org/10.1109/CVPR46437.2021.00978)
   用途：基于真实场景生成安全关键反事实变体，以及下游系统失效导向的评价思路。
   代码：准备阶段只阅读论文，不依赖其仿真栈。

6. **KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients**
   Hanselmann et al., ECCV 2022. DOI: [10.1007/978-3-031-19839-7_20](https://doi.org/10.1007/978-3-031-19839-7_20)
   用途：运动学约束、背景交通轨迹扰动和安全关键样本生成。
   代码：[autonomousvision/king](https://github.com/autonomousvision/king)，属于CARLA路线，仅借鉴约束设计。

7. **Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior (STRIVE)**
   Rempe et al., CVPR 2022. DOI: [10.1109/CVPR52688.2022.01679](https://doi.org/10.1109/CVPR52688.2022.01679)
   用途：用学习式交通先验保持生成场景真实且具有挑战性，与本项目“技能条件+学习模型+过滤”的思想最接近。
   代码：[NVlabs/STRIVE](https://github.com/NVlabs/STRIVE)，只用于理解方法和评价设计。

## 安全指标与综述

8. **A Survey of Autonomous Driving: Common Practices and Emerging Technologies**
   Yurtsever et al., IEEE Access 2020. DOI: [10.1109/ACCESS.2020.2983149](https://doi.org/10.1109/ACCESS.2020.2983149)
   用途：自动驾驶感知、预测、规划和安全评价的整体背景。

9. **Deep Learning for Safe Autonomous Driving: Current Challenges and Future Directions**
   Muhammad et al., IEEE T-ITS 2020. DOI: [10.1109/TITS.2020.3032227](https://doi.org/10.1109/TITS.2020.3032227)
   用途：长尾风险、安全验证和深度学习系统局限性。

## 阅读优先级

1. AV2论文和官方用户指南；
2. Trajectron++与VectorNet；
3. STRIVE、AdvSim和KING；
4. 两篇安全综述；
5. 进入训练阶段前再补充最新AV2轨迹预测基线。
