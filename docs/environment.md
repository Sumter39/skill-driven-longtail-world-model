# SkillDrive 环境准备

## 当前机器审计

审计日期：2026-07-19。

| 项目 | 当前状态 |
|---|---|
| WSL2 | 已安装并可启动 |
| Ubuntu | 24.04.1 LTS |
| WSL系统Python | 3.12.3，不作为项目Python |
| `uv` | 0.11.29，安装于`~/.local/bin/uv` |
| 项目Python | 3.10.20，位于Linux `.venv` |
| AV2 API | 0.3.6，导入和单场景读取已验证 |
| WSL GPU | RTX 4060 Laptop，8188 MiB，可被 `nvidia-smi` 识别 |
| D盘空间 | 随数据下载变化，使用`df -h .`实时检查 |

项目在WSL内使用Linux版Python和依赖，不与Windows Python共享虚拟环境目录。

## 安装步骤

以下命令用于新成员或重建环境；当前机器已经完成安装和同步。

在WSL Ubuntu中执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd "/mnt/d/同济大学/Course/032 大三下/大数据智能分析"
uv python install 3.10
uv python pin 3.10
uv sync --extra dev --extra av2
```

`uv sync`会自动创建项目根目录下的`.venv`。通常不需要激活环境，直接使用：

```bash
uv run python --version
uv run pytest -q
uv run python -m scripts.render_synthetic_bev
```

项目位于`/mnt/d`的NTFS文件系统，而`uv`缓存位于WSL的Linux文件系统。首次同步可能出现“Failed to hardlink files; falling back to full copy”警告。这不影响安装结果，可设置复制模式消除警告：

```bash
export UV_LINK_MODE=copy
```

如需长期使用：

```bash
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
source ~/.bashrc
```

如果需要交互式激活：

```bash
source .venv/bin/activate
```

## 快速验证

```bash
uv run python -c "import numpy, yaml, matplotlib; print('base dependencies OK')"
uv run python -c "import av2; print('AV2 API OK')"
nvidia-smi
uv run pytest -q
```

本准备阶段不要求安装PyTorch。进入模型阶段前，再根据[PyTorch官方安装页](https://pytorch.org/get-started/locally/)选择Linux和当前CUDA驱动兼容的轮子，然后检查：

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

不要在WSL内安装完整NVIDIA显卡驱动；Windows主机驱动负责向WSL提供GPU支持。

## Windows与WSL路径

| Windows | WSL2 |
|---|---|
| `D:\同济大学\Course\032 大三下\大数据智能分析\data` | `<项目根目录>/data` |
| `D:\同济大学\Course\032 大三下\大数据智能分析\outputs` | `<项目根目录>/outputs` |

默认路径由`configs/paths.example.yaml`中的仓库内相对路径定义。只有需要机器特定覆盖时才创建不提交的`configs/paths.local.yaml`。
