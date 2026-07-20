# AV2 数据下载简明指南

## 先理解要做什么

整个过程只有两次真实下载：

| 步骤 | 操作 | 是否下载 | 大约占用 |
|---|---|---|---:|
| 1 | 下载22,000个Train场景 | 是 | 5.6 GB |
| 2 | 把Train清单划分成训练、内部验证和开发子集 | 否 | 几个很小的CSV |
| 3 | 下载5,000个Validation场景 | 是 | 1.3 GB |

Train用于开发、训练和调参；Validation只用于最后评估。项目不需要Test，也不下载完整58 GB数据集。

## 0. 进入项目目录

以后每次打开WSL，先执行：

```bash
cd "/mnt/d/同济大学/Course/032 大三下/大数据智能分析"
```

## 1. 安装Windows版s5cmd

本机采用混合运行：WSL负责运行Python脚本，Windows版`s5cmd.exe`负责下载，数据写入D盘项目目录。100场景实测中，直连约0.35 MiB/s，7890代理约1.11 MiB/s，因此正式下载推荐开启FlClash并使用代理。

在Windows PowerShell执行：

```powershell
$zip = "$env:TEMP\s5cmd-windows.zip"
$install = "$HOME\.local\bin"

curl.exe --noproxy "*" -L --retry 5 `
  -o $zip `
  "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Windows-64bit.zip"

New-Item -ItemType Directory -Force $install | Out-Null
Expand-Archive -Force $zip $install
& "$install\s5cmd.exe" version
Remove-Item $zip
```

## 2. 下载Train

这是第一次真实下载。它从官方Train中固定选择22,000个场景，约占5.6 GB：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

uv run python -m scripts.data.download_av2_subset \
  --s5cmd /mnt/c/Users/123456/.local/bin/s5cmd.exe \
  --split train \
  --count 22000 \
  --seed 2026 \
  --manifest manifests/acquisition/formal_train_pool.csv \
  --execute
```

下载结果位于：

```text
data/av2/motion-forecasting/train/
```

其中：

- `--count 22000`：选择22,000个场景；
- `--seed 2026`：固定选择结果，保证以后可以复现；
- `--manifest`：把选中的场景ID写入CSV名单；
- `--execute`：真正下载。没有它就只生成名单。

每次运行分为两个独立进度阶段：第一阶段用8线程扫描清单中的本地Parquet和JSON，确认哪些场景已经完整可读；第二阶段只把不完整场景按每批200个交给Windows下载器。完整场景不会再次发送到S3检查。网络失败只重试当前批次。

## 3. 划分Train清单

Train下载完成后执行：

```bash
uv run python -m scripts.data.split_av2_train_pool
```

这一步不联网、不下载、不复制原始数据，只生成以下名单：

```text
22,000个已下载Train场景
├── 20,000个正式训练场景
│   └── 500个开发训练场景
└── 2,000个内部验证场景
    └── 100个开发验证场景
```

开发时读取500/100名单，运行更快；正式实验时切换到20,000/2,000名单。它们引用的是同一份磁盘数据。

正式训练和内部验证互不重叠，因为内部验证需要检查模型对未训练场景的泛化能力。

## 4. 下载最终Validation

这一步不用急着做。等数据读取、技能规则、模型和实验方案基本确定后再执行：

```bash
uv run python -m scripts.data.download_av2_subset \
  --s5cmd /mnt/c/Users/123456/.local/bin/s5cmd.exe \
  --split val \
  --count 5000 \
  --seed 2026 \
  --manifest manifests/splits/final_validation.csv \
  --execute
```

这是第二次真实下载，约占1.3 GB，结果位于：

```text
data/av2/motion-forecasting/val/
```

这5,000个场景只用于最终评估，不用于训练或反复调参，可以把它理解成最后考试的试卷。

## 5. 下载后检查

```bash
du -sh data/av2/motion-forecasting

find data/av2/motion-forecasting/train \
  -mindepth 1 -maxdepth 1 -type d | wc -l

find data/av2/motion-forecasting/val \
  -mindepth 1 -maxdepth 1 -type d | wc -l
```

正常情况下，Train显示22,000，Validation显示5,000。

确认数据不会上传GitHub：

```bash
git check-ignore -v data/av2/motion-forecasting/train
```

完整下载后执行全量完整性检查：

```bash
uv run python -m scripts.data.verify_av2_download \
  --manifest manifests/acquisition/formal_train_pool.csv

uv run python -m scripts.data.verify_av2_download \
  --manifest manifests/splits/final_validation.csv
```

检查会逐一确认清单中的场景文件和地图文件存在、非零，并验证Parquet页脚可读、JSON可解析。只有显示`verification passed`才视为数据准备完成。

## 6. 下载中断怎么办

直接重新执行完全相同的原下载命令。脚本默认自动复用已有CSV清单，跳过场景枚举；大小一致的完整文件会跳过，缺失或大小不一致的文件会重新下载。例如Train：

```bash
uv run python -m scripts.data.download_av2_subset \
  --s5cmd /mnt/c/Users/123456/.local/bin/s5cmd.exe \
  --split train \
  --count 22000 \
  --seed 2026 \
  --manifest manifests/acquisition/formal_train_pool.csv \
  --execute
```

不要更改split、数量或清单路径。

下载器默认使用32个并发工作线程，避免高并发压垮本地代理。代理拒绝连接、连接重置、超时、DNS失败等网络问题不会逐对象刷错误；脚本会在同一进度行提示网络中断，并每15秒无限重试，直到网络恢复或用户按`Ctrl+C`。只有权限、路径、参数等非网络错误才会打印最后一段错误并退出。

## 7. 以后数据不够再扩容

现在不需要执行本节。只有长尾场景不足时，才额外下载5,000个不重复Train场景：

```bash
uv run python -m scripts.data.download_av2_subset \
  --s5cmd /mnt/c/Users/123456/.local/bin/s5cmd.exe \
  --split train \
  --count 5000 \
  --seed 2027 \
  --exclude-manifest manifests/acquisition/formal_train_pool.csv \
  --manifest manifests/acquisition/formal_train_expansion_01.csv \
  --execute
```

## 现在应该做到哪里

当前已经完成Windows版`s5cmd.exe`安装、22,000个Train场景与5,000个Validation场景下载和全量完整性验证，并完成20,000/2,000和500/100清单划分。不下载Test，也暂不执行扩容。
