# 命令脚本目录

脚本按执行职责分组，均从仓库根目录通过 `python -m` 运行。

| 目录 | 职责 |
|---|---|
| `data/` | 下载AV2子集、划分Train池、校验本地数据完整性 |
| `seed_detection/` | 扫描技能候选池、从正式候选池筛选最终5,000个种子场景 |
| `visualization/` | 渲染AV2样例、合成BEV和候选审核图 |

示例：

```bash
uv run python -m scripts.data.verify_av2_download --help
uv run python -m scripts.seed_detection.detect_seeds --help
uv run python -m scripts.visualization.render_seed_reviews --help
```
