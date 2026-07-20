# Argoverse 2 Motion Forecasting 数据准备

## 使用范围

本项目只使用Argoverse 2的Motion Forecasting Dataset，不下载Sensor Dataset中的摄像头和激光雷达数据。

官方资料：

- [Motion Forecasting Dataset用户指南](https://argoverse.github.io/user-guide/datasets/motion_forecasting.html)
- [AV2 API代码仓库](https://github.com/argoverse/av2-api)
- [Argoverse 2论文](https://doi.org/10.48550/arXiv.2301.00493)

数据使用受Argoverse官方条款约束。原始数据、派生的大规模缓存和完整生成结果不得上传到GitHub。

## 时序约定

- 一个场景覆盖11秒；
- 采样频率为10 Hz；
- 前50步为5秒历史；
- 后60步为6秒未来；
- 正式开发前必须用一个真实样例核对`timestep`和`observed`字段，不能只依赖位置下标推断历史/未来。

## 场景文件

每个场景目录通常包含：

```text
scenario_<scenario_id>.parquet
log_map_archive_<scenario_id>.json
```

场景数据需要读取：

- `scenario_id`；
- `city_name`；
- `timestamps_ns`；
- `focal_track_id`；
- 参与者`track_id`、`object_type`和`object_category`；
- 每个状态的`timestep`、`observed`、`position`、`velocity`和`heading`。

静态地图需要读取：

- 车道左右边界；
- 由左右边界计算的中心线；
- 车道类型；
- 是否位于路口；
- 后续技能检测需要的车道拓扑关系。

当前`skilldrive.data.av2_reader`提供单场景适配接口，并已使用AV2 API官方仓库中的测试场景完成运行核验：成功读取58个参与者和213条地图折线，转换到目标参与者局部坐标系后完成BEV视觉检查。

## 目录约定

推荐本地目录：

```text
data/av2/motion-forecasting/
├── train/
├── val/
└── test/
```

路径配置：

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
```

```yaml
data_root: ./data/av2/motion-forecasting
cache_root: ./data/cache
output_root: ./outputs
```

## 下载策略

当前D盘只剩约77 GiB，因此准备阶段不下载完整数据集。

官方页面标注Motion Forecasting归档总量约58 GB。2026-07-19核对到的文件大小为：

| Split | 归档字节数 | 约合GiB |
|---|---:|---:|
| Train | 50,873,856,000 | 47.4 |
| Validation | 6,364,764,160 | 5.9 |
| Test | 4,447,068,160 | 4.1 |
| 合计 | 61,685,688,320 | 57.5 |

不建议在当前D盘下载`.tar`后再解包，因为归档和解包文件会短期占用双份空间。官方推荐使用`s5cmd`直接复制S3中的散文件，不需要AWS账号，也不产生额外解包副本。

### 安装Windows版s5cmd

本机采用混合方案：WSL运行Python脚本，Windows版`s5cmd.exe`负责S3查询和下载，数据直接写入D盘项目目录。安装包可通过开启代理的Windows浏览器下载：

```text
https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Windows-64bit.zip
```

在Windows PowerShell解压安装：

```powershell
$zip = "$HOME\Downloads\s5cmd_2.3.0_Windows-64bit.zip"
$install = "$HOME\.local\bin"
New-Item -ItemType Directory -Force $install | Out-Null
Expand-Archive -Force $zip $install
& "$install\s5cmd.exe" version
```

### 不下载完整 split

课程开发不需要 Test split，也不计划下载完整 Train 和 Validation。不要直接对官方 split 使用通配符复制，否则会下载约 58 GB 的完整数据。项目统一通过确定性子集脚本下载，完整命令见[数据下载命令](download-commands.md)。

### 推荐的确定性子集

按归档平均大小估算，每个场景及其局部地图约250 KB。项目采用：

| 用途 | 来源 | 场景数 | 估算空间 |
|---|---|---:|---:|
| 正式训练和内部验证下载池 | Train | 22,000 | 约5.6 GB |
| 最终验证子集 | Validation | 5,000 | 约1.3 GB |

总量约7 GB。实际大小以下载结果为准。如果经确认的30类技能存在真实种子覆盖不足，再以5,000个Train场景为一批讨论是否增量扩展。

`scripts/data/download_av2_subset.py`先使用`s5cmd ls`列出官方场景，通过固定随机种子选择ID，写出可提交的CSV清单，然后可选执行下载。S3对象列表缓存在被Git忽略的`data/metadata/`。

正式Train池下载和清单划分示例：

```bash
uv run python -m scripts.data.download_av2_subset \
  --s5cmd /mnt/c/Users/123456/.local/bin/s5cmd.exe \
  --split train \
  --count 22000 \
  --seed 2026 \
  --manifest manifests/acquisition/formal_train_pool.csv \
  --execute

uv run python -m scripts.data.split_av2_train_pool
```

划分脚本不下载或复制数据。它将22,000个场景固定划分为20,000个正式训练场景和2,000个内部验证场景，并从两者分别派生500/100开发清单。正式训练与内部验证按`scenario_id`互斥；两个开发清单则分别是它们的子集。

最终Validation下载示例：

```bash
uv run python -m scripts.data.download_av2_subset \
  --s5cmd /mnt/c/Users/123456/.local/bin/s5cmd.exe \
  --split val \
  --count 5000 \
  --seed 2026 \
  --manifest manifests/splits/final_validation.csv \
  --execute
```

命令默认下载到仓库内的`data/av2/motion-forecasting`。`--s5cmd`显式指定Windows下载器；WSL脚本会自动把命令文件和目标目录转换为Windows路径。清单不存在时自动创建，已存在时自动复用并续传；只有需要重新抽样时才显式传入`--force-manifest`。完整命令见[数据下载命令](download-commands.md)。

执行原则：

1. 先阅读官方用户指南和许可条款；
2. 使用官方小型测试场景验证接口；
3. 使用固定随机种子下载22,000个Train场景，并从其清单派生500/100开发子集；
4. 实验方案确定后下载5,000个官方Validation场景用于最终评估；
5. 处理完成后只保留必要的矢量特征和场景清单；
6. 不同时保留完整原始数据、全部缓存和全部BEV渲染结果。

具体下载命令应以官方用户指南当时给出的S3路径和工具为准，不在仓库中硬编码可能变化的下载地址。

### 官方小型测试场景

前期接口验证使用AV2 API官方GitHub仓库中的测试数据，不属于正式训练数据：

```bash
uv run python -m scripts.data.download_av2_test_sample
```

该脚本只下载一个场景的parquet和地图JSON，总大小约220 KB，并校验文件大小。文件保存在被Git忽略的：

```text
data/sample/av2/0a1e6f0a-1817-4a98-b02e-db8c9327d151/
```

渲染命令：

```bash
uv run python -m scripts.visualization.render_av2_sample \
  data/sample/av2/0a1e6f0a-1817-4a98-b02e-db8c9327d151/scenario_0a1e6f0a-1817-4a98-b02e-db8c9327d151.parquet
```

## 场景清单

CSV列顺序固定为：

```text
scenario_id,split,source_path,city_name,selected_reason
```

示例：

```csv
scenario_id,split,source_path,city_name,selected_reason
example-id,development,data/av2/example/scenario_example-id.parquet,MIA,reader-smoke-test
```

`skilldrive.data.manifests.assert_disjoint`负责检查训练、验证和测试清单是否存在重复`scenario_id`。

## 单场景验证清单

取得合法样例后必须检查：

- parquet和地图JSON可以加载；
- `focal_track_id`能够找到对应轨迹；
- 历史和未来掩码符合50/60步约定；
- 缺失状态保留为NaN和无效掩码，不填成零坐标；
- 地图边界、中心线和轨迹处于同一全局坐标系；
- 转换到局部坐标后，目标参与者位于原点附近且朝向局部x轴；
- BEV图中地图和轨迹没有明显错位。
