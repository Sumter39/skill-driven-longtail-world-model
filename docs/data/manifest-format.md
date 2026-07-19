# 场景清单与防泄漏规则

## 字段

| 字段 | 含义 |
|---|---|
| `scenario_id` | AV2场景唯一标识 |
| `split` | `development`、`train`、`validation`或`test` |
| `source_path` | 场景parquet路径 |
| `city_name` | 场景城市 |
| `selected_reason` | 入选原因或对应技能种子说明 |

## 不变量

- 一个清单内`scenario_id`必须唯一。
- 训练、验证和测试清单必须两两无交集。
- 未来生成样本继承源`scenario_id`，并额外记录变体ID，不能被重新划入验证集。
- 数据划分先于技能检测和生成，禁止按照实验结果重新选择测试场景。
- 清单、筛选配置和随机种子应提交Git；原始数据路径可使用相对根目录表达。

## 接口

```python
from skilldrive.data.manifests import assert_disjoint, read_manifest

train = read_manifest("manifests/train.csv")
validation = read_manifest("manifests/validation.csv")
assert_disjoint(train, validation)
```
