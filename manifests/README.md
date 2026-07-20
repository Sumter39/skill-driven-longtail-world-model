# 场景清单目录

这里只保存需要版本控制、可复现数据划分或最终选择结果的CSV清单。原始数据、扫描checkpoint和可再生成的候选池放在被Git忽略的 `data/` 或 `outputs/` 中。

| 目录 | 职责 |
|---|---|
| `acquisition/` | 已下载的AV2池及未来增量下载清单 |
| `splits/` | 20,000正式Train、2,000内部验证和5,000最终Validation |
| `development/` | 从正式划分固定派生的500/100开发子集 |
| `seeds/` | 最终确定性种子清单；不保存原始候选池 |

关键约束：

- `formal_train.csv`、`internal_validation.csv`和`final_validation.csv`两两零交集；
- `development_train.csv`是正式Train的固定子集；
- `development_validation.csv`是内部验证的固定子集；
- 正式候选池保存在 `outputs/seed_detection/formal_candidate_pool.csv`，最终5,000场景才写入 `seeds/formal_candidates.csv`。
