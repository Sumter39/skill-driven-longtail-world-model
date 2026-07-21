# SkillDrive 环境准备

## 当前机器审计

审计日期：2026-07-21。

| 项目 | 当前状态 |
|---|---|
| WSL2 | 已安装并可启动 |
| Ubuntu | 24.04.1 LTS |
| WSL系统Python | 3.12.3，不作为项目Python |
| `uv` | 0.11.29，安装于`~/.local/bin/uv` |
| 项目Python | 3.10.20，位于Linux `.venv` |
| AV2 API | 0.3.6，导入和单场景读取已验证 |
| WSL GPU | RTX 4060 Laptop，8188 MiB，可被 `nvidia-smi` 识别 |
| PyTorch | 2.13.0+cu130，CUDA前向和反向已验证 |
| GPU驱动 | Windows 610.74，WSL暴露CUDA 13.3能力 |
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
uv sync
```

`uv sync`会自动创建项目根目录下的`.venv`。通常不需要激活环境，直接使用：

```bash
uv run python --version
uv run pytest -q
uv run python -m scripts.visualization.render_synthetic_bev
```

`av2`和`torch==2.13.0`是项目正式依赖；`pytest`位于`dependency-groups.dev`，并由`tool.uv.default-groups`默认启用。因此普通`uv sync`和`uv run`都会保留数据、GPU训练和测试依赖，不需要`--extra`或`--all-extras`。

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
uv run python -c "import numpy, yaml, matplotlib, torch; print('base dependencies OK')"
uv run python -c "import av2; print('AV2 API OK')"
nvidia-smi
uv run pytest -q
```

当前锁文件固定PyTorch 2.13.0，Linux轮子使用CUDA 13.0运行时；Windows驱动向WSL提供的CUDA 13.3能力向后兼容。完整GPU验证为：

```bash
uv run python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
x.square().mean().backward()
print(float(x.grad.abs().mean()))
PY
```

不要在WSL内安装完整NVIDIA显卡驱动或系统级CUDA Toolkit；项目轮子已携带所需CUDA运行库，Windows主机驱动负责向WSL提供GPU支持。

## Windows与WSL路径

| Windows | WSL2 |
|---|---|
| `D:\同济大学\Course\032 大三下\大数据智能分析\data` | `<项目根目录>/data` |
| `D:\同济大学\Course\032 大三下\大数据智能分析\outputs` | `<项目根目录>/outputs` |

默认路径由`configs/paths.example.yaml`中的仓库内相对路径定义。只有需要机器特定覆盖时才创建不提交的`configs/paths.local.yaml`。
