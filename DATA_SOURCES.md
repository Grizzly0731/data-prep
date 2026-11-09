# 数据来源与版本记录

记录每份数据集的**来源链接 / 仓库名称、版本号、获取日期、原始文件信息**，用于来源核对与版本管理。

| 项 | 内容 |
| --- | --- |
| 学号 / 姓名 | 2025110789 / 廖瑾宇 |
| 统一获取日期 | 2026-11-09 |
| 记录文件 | `DATA_SOURCES.md`（本文件） |
| 核对脚本 | `prepare_data.py`（末尾打印输出文件 SHA-256） |

---

## 一、来源记录总表

| 数据集 | 版本 | 来源链接 / 仓库 | 获取日期 | 原始文件 | 原始大小 | 最终输出 |
| --- | --- | --- | --- | --- | --- | --- |
| Iris 鸢尾花 | 课程指定版（150 样本） | 课程提供的 `data/iris_dataset.csv` | 2026-11-09 | `iris_dataset.csv` | 4033 B | 3 个 CSV |
| MNIST | 官方 IDX 版（60000 + 10000） | https://storage.googleapis.com/cvdf-datasets/mnist/ | 2026-11-09 | 4 个 `.gz` | 合计约 11.6 MB | mnist_train / mnist_test |
| 波士顿房价 | UCI 原版（506 × 14） | https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data | 2026-11-09 | `housing.data` | 49082 B | boston_housing.csv |
| 泰坦尼克号 | Kaggle `train.csv` 原版（891 × 12） | https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv | 2026-11-09 | `titanic.csv` | 60302 B | titanic_train.csv |
| 西瓜数据集 3.0 | **3.0**（17 样本） | 周志华《机器学习》表 4.1，无在线链接 | 2026-11-09 | 无电子文件（脚本内嵌） | — | watermelon3.0.csv |

---

## 二、逐项说明

### 1. Iris 鸢尾花

- **来源**：课程提供的本地文件 `data/iris_dataset.csv`
- **版本**：课程指定版本，150 行、4 特征 3 类别
- **获取日期**：2026-11-09
- **原始文件**：`iris_dataset.csv`，4033 字节，UTF-8 带 BOM
- **一致性核对**：文件首行应为 `sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),target_name`

### 2. MNIST

- **来源**：Google CVDF 托管的 MNIST 官方 IDX 文件镜像
  `https://storage.googleapis.com/cvdf-datasets/mnist/`
- **版本**：官方 IDX 版本（训练 60000、测试 10000，28 × 28）
- **获取日期**：2026-11-09
- **原始文件与大小**

  | 文件 | 大小 |
  | --- | --- |
  | `train-images-idx3-ubyte.gz` | 9912422 B |
  | `train-labels-idx1-ubyte.gz` | 28881 B |
  | `t10k-images-idx3-ubyte.gz` | 1648877 B |
  | `t10k-labels-idx1-ubyte.gz` | 4542 B |

- **一致性核对**：`struct` 读出的 magic 必须为 2051（图像）/ 2049（标签）；训练集 60000 行、测试集 10000 行

### 3. 波士顿房价

- **来源**：UCI Machine Learning Repository
  `https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data`
- **版本**：UCI 原版，506 行 × 14 列
- **获取日期**：2026-11-09
- **原始文件**：`housing.data`，49082 字节，空白分隔、无表头
- **一致性核对**：去空行后必须恰好 506 行 × 14 列；首行应为 `0.00632 18.00 2.310 0 0.5380 6.5750 65.20 4.0900 1 296.0 15.30 396.90 4.98 24.00`
- **说明**：未使用已弃用的 `sklearn.datasets.load_boston`

### 4. 泰坦尼克号（全组统一来源）

- **固定来源**：`https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`
  （Kaggle `titanic/train.csv` 的公开镜像）
- **版本**：Kaggle `train.csv` 原版，891 行 × 12 列
- **获取日期**：2026-11-09
- **原始文件**：`titanic.csv`，60302 字节，UTF-8 带表头
- **一致性核对**
  - 行数 891、列数 12
  - 列名顺序：`PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked`
  - 缺失数：Age 177、Cabin 687、Embarked 2
- **为什么固定同一来源**：Kaggle 上有多个改名、改行的衍生版本（如删掉 Cabin、把 Sex 转成 0/1），不同同学各取一份会导致行数、列名、缺失数全部对不上，无法互相对比。本项目统一使用上面这一个 URL，**不替换、不改名、不删列**。

### 5. 西瓜数据集 3.0

- **来源**：周志华《机器学习》（清华大学出版社，2016）第 4 章**表 4.1「西瓜数据集 3.0」**
- **版本**：**3.0**（17 个样本：好瓜 8、坏瓜 9；6 离散 + 2 连续特征）
- **获取日期**：2026-11-09
- **原始文件**：无在线文件，数据以 `ROWS` 常量形式内嵌在 `prepare_watermelon.py` 中
- **一致性核对**：17 行 × 9 列；好瓜 8 个、否 9 个

> **版本注意**：西瓜数据集另有 **2.0**（17 样本但只有离散特征，无密度/含糖率）和 **4.0**（30 样本，含「纹理」等多值属性）。本项目仅使用 **3.0**，文件名固定为 `watermelon3.0.csv`；若要引入 4.0，必须单独命名并说明用途，不得与本文件混用。

---

## 三、一致性核对方法

### 3.1 来源一致性

| 检查项 | 期望值 | 如何核对 |
| --- | --- | --- |
| 泰坦尼克原始文件大小 | 60302 字节 | `Get-Item data\titanic_raw\titanic.csv` |
| 波士顿原始文件大小 | 49082 字节 | `Get-Item data\boston_raw\housing.data` |
| MNIST 训练图像 | 9912422 字节 | `Get-Item data\mnist_raw\train-images-idx3-ubyte.gz` |
| 各数据集行列数 | 见 README | `python validate_data.py` |

### 3.2 转换结果一致性

`prepare_data.py` 结束时会打印每个输出文件的 SHA-256（前 16 位）：

```
train_scaled.csv                 104 x 5      9.1 KB  ffeae57d74f7f01f
test_scaled.csv                   45 x 5      4.0 KB  16e6b0c2b002e24e
scaler_params.csv                  4 x 3      0.2 KB  8dbdcec911c54aa5
mnist_train.csv              60000 x 785   104.56 MB  6a3ff6ff8f9baaaa
mnist_test.csv               10000 x 785    17.45 MB  1638ea507de03904
boston_housing.csv              506 x 14     36.8 KB  bd6afa0444d4bc30
titanic_train.csv               891 x 12     61.4 KB  f77b80f5d4d8b6e3
watermelon3.0.csv                 17 x 9      1.0 KB  edb8a01d83acc0b1
```

**同一个来源 + 同一份脚本 ⇒ 必须得到同一串哈希。** 若哈希不同，说明来源文件被换过或脚本被改过。

---

## 四、版本变更记录

| 日期 | 变更内容 |
| --- | --- |
| 2026-11-09 | 建立项目目录；完成 5 份数据的转换脚本；固定泰坦尼克号为 datasciencedojo 镜像 |
| 2026-11-09 | 补充 `validate_data.py`（9 个 CSV 校验）与 `check_csv.py`（编码/分隔符规范检查） |
| 2026-11-09 | 建立本文件，记录来源、版本、日期与一致性核对方法 |

---

## 五、声明

以上来源与版本信息均经本人核对；数据未做与来源不符的修改。若后续更换来源或版本，将在此表追加记录，不覆盖旧记录。

签名：廖瑾宇　　日期：2026-11-09
