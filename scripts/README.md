# 命令脚本目录

脚本按执行职责分组，均从仓库根目录通过 `python -m` 运行。

| 目录 | 职责 |
|---|---|
| `data/` | 下载AV2子集、划分Train池、校验本地数据完整性 |
| `seed_detection/` | 默认全量或按`--skill-id`扫描34类正式技能；验证并合并历史39规则扫描的技能互斥checkpoint、排除5类已验证零命中候选规则，并筛选最终5,000个种子场景 |
| `visualization/` | 渲染AV2样例、合成BEV和候选审核图 |

示例：

```bash
uv run python -m scripts.data.verify_av2_download --help
uv run python -m scripts.seed_detection.detect_seeds --help
uv run python -m scripts.seed_detection.merge_candidate_pools --help
uv run python -m scripts.visualization.render_seed_reviews --help
```

`merge_candidate_pools`中的`--exclude-skill-id`只用于从正式候选池排除已经验证为零命中的候选规则；39规则历史扫描来源仍保留在`outputs/seed_detection/expanded_39/`中。
