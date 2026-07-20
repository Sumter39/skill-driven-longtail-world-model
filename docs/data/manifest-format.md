# 场景清单与防泄漏规则

## 目录职责

- `manifests/acquisition/`：下载池和增量下载清单；
- `manifests/splits/`：正式训练、内部验证和最终Validation；
- `manifests/development/`：固定开发子集；
- `manifests/seeds/`：最终确定性种子清单。

开发和正式扫描产生的原始候选池、summary与checkpoint均放在被Git忽略的 `outputs/seed_detection/`，不混入场景划分清单。

## 字段

| 字段 | 含义 |
|---|---|
| `scenario_id` | AV2场景唯一标识 |
| `split` | `development_train`、`development_validation`、`train`、`internal_validation`或`validation` |
| `source_path` | 场景parquet路径 |
| `city_name` | 场景城市 |
| `selected_reason` | 入选原因或对应技能种子说明 |

## 不变量

- 一个清单内`scenario_id`必须唯一。
- `formal_train`、`internal_validation`和`final_validation`必须按`scenario_id`两两无交集。
- `development_train`是`formal_train`的固定子集，`development_validation`是`internal_validation`的固定子集；这种开发清单与父清单的重叠是有意设计，不属于数据泄漏。
- 未来生成样本继承源`scenario_id`，并额外记录变体ID，不能被重新划入验证集。
- 数据划分先于技能检测和生成，禁止按照实验结果重新选择测试场景。
- 清单、筛选配置和随机种子应提交Git；原始数据路径可使用相对根目录表达。

## 接口

```python
from skilldrive.data.manifests import assert_disjoint, read_manifest

train = read_manifest("manifests/splits/formal_train.csv")
internal_validation = read_manifest("manifests/splits/internal_validation.csv")
final_validation = read_manifest("manifests/splits/final_validation.csv")
assert_disjoint(train, internal_validation, final_validation)
```
