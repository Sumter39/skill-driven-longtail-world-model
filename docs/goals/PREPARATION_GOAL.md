# SkillDrive 前期准备 Goal

## 1. 当前目标

本阶段只完成 SkillDrive 项目的前期准备，为后续正式开发和训练建立可靠基础。

技术路线暂定为：

> Argoverse 2 Motion Forecasting + 条件 CVAE 轨迹世界模型 + BEV 场景表达 + 轻量矢量 Transformer 轨迹预测验证。

本阶段禁止启动任何模型训练、批量轨迹生成或正式对比实验。即使环境和代码提前准备完成，也必须在用户明确确认后才能进入训练阶段。

## 2. 本阶段范围

### 2.1 允许完成的工作

- 检查和整理 Git 仓库结构。
- 编写环境配置、依赖说明和安装文档。
- 准备 WSL2 Ubuntu 24.04、`uv`、Python 3.10 和 CUDA 的环境方案。
- 调研并记录 Argoverse 2 Motion Forecasting 的数据结构、下载方式和许可要求。
- 编写数据目录约定、路径配置和场景清单格式。
- 实现或设计单场景数据读取接口。
- 使用一个样例或最小数据验证坐标系、轨迹字段和 HD Map 字段。
- 完成单场景 BEV 静态图的最小可视化闭环。
- 定义统一的场景、参与者、地图、技能和过滤结果数据结构。
- 建立 30 类技能目录，并为 5 类核心技能编写详细规则规范。
- 为数据读取、坐标变换、技能配置和 BEV 可视化建立测试骨架。
- 整理相关论文、开源项目、数据集文档和技术风险。
- 更新 README，使仓库准确反映当前项目状态和下一阶段入口。

### 2.2 明确禁止的工作

- 不训练条件 CVAE、Transformer、LSTM 或其他神经网络。
- 不下载或加载预训练模型权重进行正式推理。
- 不启动长时间 GPU 任务。
- 不批量生成长尾场景。
- 不处理完整数据集或建立大规模特征缓存。
- 不运行正式 E0–E3 对比实验或消融实验。
- 不创建大量模型权重、日志、视频或中间产物。
- 不提交或推送 Git，除非用户另行明确要求。
- 不改变已经确定的技术路线；遇到重大分歧时先记录并向用户确认。

## 3. 关键技术结论

### 3.1 BEV 的角色

BEV 不作为像素生成模型，而作为统一的俯视场景表达、质量审核和结果展示接口。

后续世界模型生成的是关键参与者的未来轨迹。BEV 负责绘制：

- 车道中心线和道路边界；
- 参与者历史轨迹；
- 真实未来轨迹；
- 后续生成的反事实未来轨迹；
- 风险参与者、技能标签和安全指标。

### 3.2 世界模型路线

后续正式阶段使用条件 CVAE 作为轨迹世界模型。技能规则负责提供触发条件、参与者角色、连续参数、交通约束和风险目标；未来轨迹必须由学习模型生成，不能退化为简单坐标平移。

### 3.3 下游验证路线

后续正式阶段使用轻量矢量 Transformer 作为主要轨迹预测器，并保留恒速模型和 LSTM 作为基础基线。本阶段只定义接口和配置，不实现训练流程。

## 4. 前期仓库结构

准备阶段应建立以下最小结构，但只在实际需要时创建文件，禁止生成空目录或无用途占位代码：

```text
skill-driven-longtail-world-model/
├── configs/
│   ├── paths.example.yaml
│   ├── data/
│   └── skills/
├── docs/
│   ├── goals/
│   ├── data/
│   └── references/
├── skilldrive/
│   ├── data/
│   ├── schemas/
│   ├── skills/
│   └── visualization/
├── tests/
├── .gitignore
└── README.md
```

本阶段不创建 `models/`、`training/`、`checkpoints/` 或正式实验目录，避免暗示已经进入训练阶段。

## 5. 环境准备

### 5.1 目标环境

- WSL2 Ubuntu 24.04；
- 使用 `uv` 安装和管理 Python 3.10；
- 使用项目根目录下的 `.venv` 隔离依赖；
- Miniforge/Conda 仅作为团队成员无法使用 `uv` 时的备用方案；
- 与 RTX 4060 Laptop 兼容的 CUDA 和 PyTorch；
- 本机 GPU 显存约 8 GB；
- 数据和缓存以 D 盘剩余空间为约束，采用分片处理方案。

### 5.2 环境管理原则

- Windows 和 WSL2 使用不同操作系统二进制，不能共享同一个 Conda 或虚拟环境目录。
- Windows 中已经安装的 Anaconda 不作为 WSL2 项目环境。
- WSL2 内由 `uv` 单独安装 Python 3.10，并在仓库根目录创建 `.venv`。
- 团队共享 `pyproject.toml` 和依赖说明，不共享 `.venv` 目录。
- `.venv` 必须由 `.gitignore` 排除。
- 不使用 Ubuntu 系统 Python 3.12 直接运行项目，避免与 AV2 和研究代码的兼容性风险。
- 不在 WSL2 中安装完整 NVIDIA 显卡驱动；GPU 由 Windows 主机驱动提供，WSL2 只安装对应的 Linux PyTorch 包。

### 5.3 `uv` 安装与环境创建

在 WSL2 Ubuntu 终端中执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd "/mnt/d/同济大学/Course/032 大三下/大数据智能分析"
uv python install 3.10
uv python pin 3.10
uv sync --extra dev --extra av2
```

说明：

- `uv python install 3.10` 安装由 `uv` 管理的 Linux Python，不修改 Ubuntu 系统 Python。
- `uv python pin 3.10` 创建可提交的 `.python-version`，让团队统一使用 Python 3.10。
- `uv sync --extra dev --extra av2` 根据 `pyproject.toml` 和 `uv.lock` 自动创建或更新仓库根目录下的 `.venv`，并安装测试依赖和 AV2 API。
- 日常使用 `uv run <command>`，无需手工激活 `.venv`；需要交互式终端时仍可执行 `source .venv/bin/activate`。
- `.venv` 必须忽略，`.python-version` 和 `uv.lock` 应提交。
- 项目位于`/mnt/d`时，`uv`缓存与`.venv`跨文件系统，允许设置`UV_LINK_MODE=copy`；硬链接降级警告不是安装失败。
- 当前准备阶段不强制安装 PyTorch；如果需要验证 GPU 导入，应根据 PyTorch 官方 Linux/CUDA 安装页选择与当前驱动兼容的轮子，不凭经验硬编码 CUDA 版本。

### 5.4 快速验证

环境创建后只允许执行快速检查：

```bash
uv run python --version
uv run python -c "import numpy, yaml, matplotlib; print('base dependencies OK')"
uv run python -c "import av2; print('AV2 API OK')"
nvidia-smi
uv run pytest -q
```

如果后续安装了 PyTorch，可以额外执行：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

该命令只能检查导入和 GPU 可见性，不得在准备阶段启动训练或大规模张量计算。

### 5.5 环境文件

准备阶段应提供：

- `pyproject.toml` 作为 `uv` 的主要项目和依赖定义；
- 可选的 `environment.yml`，仅供偏好 Conda 的团队成员备用；
- 依赖版本说明；
- `uv` 安装、Python 3.10 固定、依赖同步和 `uv run` 使用命令；
- WSL2 中的 GPU 可见性检查命令；
- Python、PyTorch、CUDA 和 AV2 工具链的验证命令；
- Windows 路径与 WSL2 路径的对应说明。

环境验证只能执行快速导入、单元测试和设备检查，不进行模型训练。

## 6. 数据准备规范

### 6.1 数据来源

只考虑 Argoverse 2 Motion Forecasting 数据，不下载摄像头和激光雷达数据。

准备文档应记录：

- 官方下载地址和访问方式；
- 数据许可及不可上传到 GitHub 的内容；
- 训练集、验证集和测试集的用途；
- 场景文件、参与者轨迹和静态地图的主要字段；
- 5 秒历史、6 秒未来、10 Hz 时间序列约定；
- 本地数据目录和 Git 忽略规则。

### 6.2 路径约定

代码不得硬编码绝对路径。默认使用不提交的本地配置：

```yaml
data_root: /mnt/d/datasets/av2/motion-forecasting
cache_root: /mnt/d/skilldrive-cache
output_root: /mnt/d/skilldrive-outputs
```

仓库只提交 `paths.example.yaml`，真实路径文件使用 `.gitignore` 排除。

### 6.3 场景清单

为后续防止数据泄漏，先定义场景清单格式：

```text
scenario_id,split,source_path,city_name,selected_reason
```

本阶段只用极少量样例建立开发清单，不扫描完整训练集。

后续数据下载采用确定性子集，而不是完整下载：先取500个Train和100个Validation用于开发；正式阶段固定下载22,000个Train和5,000个Validation，并在技能种子不足时按5,000个场景递增。

## 7. 数据结构接口

准备阶段需要确定以下公开数据结构，优先使用 Python `dataclass`，避免过早引入复杂框架。

### Scenario

- `scenario_id`
- `city_name`
- `timestamps`
- `focal_track_id`
- `agents`
- `map_polylines`
- `metadata`

### AgentTrack

- `track_id`
- `object_type`
- `positions`
- `velocities`
- `headings`
- `observed_mask`
- `is_focal`

### MapPolyline

- `polyline_id`
- `polyline_type`
- `points`
- `direction`
- `is_intersection`

### SkillSpec

- `skill_id`
- `family`
- `implemented`
- `trigger`
- `actors`
- `parameters`
- `constraints`
- `risk_definition`
- `expected_behavior`
- `output_labels`

### FilterReport

- `passed`
- `hard_failures`
- `component_scores`
- `total_score`
- `risk_metrics`

本阶段可以定义结构、解析和序列化测试，但不实现模型输入张量和训练批处理。

## 8. 技能目录准备

30 类技能分为六组：

- 车辆交互：切入、急减速、跟停、汇入、无保护转向冲突。
- 弱势参与者：行人横穿、骑行者横穿、逆行骑行、路侧突然进入、群体横穿。
- 道路障碍：临停车辆、遗落物、低矮障碍、车道变窄、车道阻塞。
- 施工变化：施工绕行、临时封道、锥桶导流、临时改道、临时限速。
- 环境传感器：降雨、浓雾、眩光、低照度、相机退化。
- 规则优先权：未让行、闯灯、应急车辆、四向停车歧义、无控制路口抢行。

需要为下列五类核心技能写出完整规则规范，但本阶段不批量执行：

1. 相邻车辆切入；
2. 前车急减速；
3. 慢车或停车阻塞；
4. 行人或骑行者横穿；
5. 汇入或路口让行。

每个核心技能必须明确：

- 种子场景触发条件；
- 风险发起者和受影响者；
- 可控连续参数；
- 地图和运动学约束；
- TTC 或其他风险定义；
- 期望驾驶行为；
- 未来过滤指标；
- AV2 数据能否直接表达，不能表达的内容如何降级处理。

## 9. 单场景 BEV 最小闭环

本阶段唯一允许的场景级运行目标是：

```text
读取一个AV2样例场景
        ↓
解析参与者轨迹和静态地图
        ↓
转换到目标参与者局部坐标系
        ↓
绘制历史轨迹、真实未来和地图
        ↓
保存一张BEV检查图
```

BEV 图至少包含：

- 目标参与者；
- 邻近车辆、行人和骑行者；
- 历史轨迹与真实未来轨迹的不同线型；
- 车道中心线和道路边界；
- 图例、坐标方向、场景 ID 和时间范围。

这一步只验证数据理解和表示方式，不包含生成轨迹。

## 10. 前期测试清单

必须准备并通过：

- 配置文件能够覆盖默认路径且不提交真实本地路径；
- 数据结构能够完成序列化和反序列化；
- 全局坐标到局部坐标再回到全局坐标的误差小于 `1e-4`；
- 旋转、平移和航向角转换正确；
- 变长参与者轨迹能够通过掩码表示；
- 缺失轨迹点不会被误当成零坐标；
- 技能 YAML 的必填字段校验正确；
- 五类核心技能分别具有有效配置样例；
- BEV 绘图函数能在无显示器环境下保存图片；
- 训练集和验证集的场景清单接口能够检查 ID 交集。

所有测试只能使用合成数据或极少量样例数据，不触发模型训练。

## 11. 调研与参考资料

准备阶段需要形成简洁的参考资料清单，至少覆盖：

- Argoverse 2 Motion Forecasting 数据和地图 API；
- 多智能体轨迹预测基本方法；
- CVAE 轨迹生成；
- 矢量地图和 Transformer 场景编码；
- 反事实驾驶场景生成；
- 运动学、地图、碰撞和 TTC 质量检查；
- BEV 轨迹可视化。

每条参考资料记录标题、链接、用途和是否有可复用代码，不在本阶段复现论文模型。

## 12. 前期准备验收标准

只有同时满足以下条件，才能向用户申请进入训练阶段：

- 仓库结构简洁，没有无用途的占位模块。
- WSL2、`uv`、Python 3.10 和基础依赖快速检查通过；如果安装了 PyTorch，还需完成 GPU 可见性检查。
- AV2数据下载和目录说明完整。
- 确定性子集下载脚本和固定随机种子策略完整。
- 至少一个样例场景能够读取并生成正确BEV图。
- 坐标转换和数据结构测试全部通过。
- 30类技能目录完成，5类核心技能规范完整。
- 训练/验证场景清单格式和防泄漏检查完成。
- README准确说明当前只完成前期准备。
- 已整理环境、数据、技能和技术风险清单。
- 没有启动任何模型训练、批量生成或正式实验。

## 13. Goal 模式执行规则

Goal 模式按以下顺序执行：

1. 阅读本文件和 `FULL_PROJECT_PLAN.md`，但只执行本文件范围。
2. 检查当前仓库和已有文件，不重复创建已有内容。
3. 先完成环境与数据调研文档，再建立最小代码结构。
4. 使用合成数据编写并验证坐标转换和数据结构测试。
5. 只有在具备合法样例数据时，才执行单场景 AV2 读取和 BEV 绘制。
6. 如果缺少数据、WSL2或依赖，记录明确阻塞和手动操作步骤，不擅自扩大下载或安装范围。
7. 禁止训练、批量生成和正式实验。
8. 禁止提交或推送 Git，除非用户另行明确授权。
9. 完成后输出前期准备清单、测试结果、剩余阻塞和进入训练阶段前的建议。

## 14. 当前执行状态

截至2026-07-19，本文件范围内的环境、代码骨架、技能目录、合成测试和官方AV2小型样例验证均已完成。详细证据见`../preparation-status.md`。

仍然禁止进入模型训练。进入下一阶段前只剩项目范围外的决策门槛：确认正式数据存储空间、安装GPU版PyTorch并获得用户明确授权。
