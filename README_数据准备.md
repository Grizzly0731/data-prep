# 2025110789_廖瑾宇_数据准备

数据准备阶段的工程目录，用于读取原始数据、检查数据质量、划分数据集并完成标准化。

## 目录结构

```
2025110789_廖瑾宇_数据准备/
├── data/
│   ├── iris_dataset.csv      原始数据（150 行 × 5 列）
│   ├── train_scaled.csv      训练集（标准化后，脚本生成）
│   ├── test_scaled.csv       测试集（标准化后，脚本生成）
│   ├── scaler_params.csv     标准化参数（均值、标准差，脚本生成）
│   ├── mnist_train.csv       MNIST 训练集（60000 × 785，脚本生成）
│   ├── mnist_test.csv        MNIST 测试集（10000 × 785，脚本生成）
│   ├── boston_housing.csv    波士顿房价（506 × 14，脚本生成）
│   ├── titanic_train.csv     泰坦尼克号（891 × 12，脚本生成）
│   └── watermelon3.0.csv     西瓜数据集 3.0（17 × 9，脚本生成）
├── prepare_data.py           总入口：打印来源登记表 → 依次调用各转换脚本 → 输出校验
├── prepare_iris.py           Iris 转换脚本
├── prepare_mnist.py          MNIST 转换脚本
├── prepare_boston.py         波士顿房价转换脚本
├── prepare_titanic.py        泰坦尼克号转换脚本
├── prepare_watermelon.py     西瓜数据集 3.0 转换脚本
├── check_csv.py              CSV 规范检查脚本（编码/分隔符/索引列）
├── validate_data.py          数据校验与读取测试（行数/列名/缺失值）
└── README.md                 本说明文件
```

## 运行方式

在项目目录下打开终端，一条命令完成全部转换：

```bash
python prepare_data.py
```

只转换其中几个：

```bash
python prepare_data.py iris titanic
```

依赖：`pandas`。MNIST 与波士顿房价、泰坦尼克号需要联网下载（已下载过会自动跳过）。

## 数据来源登记表

`prepare_data.py` 会先打印这张表，实现「转换过程可追溯」：

| 数据集 | 原始来源 | 转换方式 | 输出 |
| --- | --- | --- | --- |
| Iris | `data/iris_dataset.csv`（课程提供，150 × 5） | pandas 读取 → 查缺失/重复 → 7:3 划分 → 训练集 Z-score | train_scaled / test_scaled / scaler_params |
| MNIST | `storage.googleapis.com/cvdf-datasets/mnist/` 的 4 个 IDX `.gz` | gzip + struct 解析文件头 → NumPy 展平 784 → 无表头保存 | mnist_train / mnist_test |
| 波士顿房价 | UCI `housing.data` | 空白分隔文本 → `sep=r"\s+"` → 补 14 个列名 | boston_housing |
| 泰坦尼克号 | Kaggle `train.csv` 镜像 | pandas 读取 → 校验 891×12 → 原样保存 | titanic_train |
| 西瓜 3.0 | 周志华《机器学习》表 4.1 | 内嵌录入 → 校验字段值域 → UTF-8-SIG | watermelon3.0 |

## 可复现性验证

脚本运行结束后会打印每个输出文件的**行列数与 SHA-256（前 16 位）**。连续运行两次，哈希值完全相同：

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

## Iris 转换细节

`prepare_iris.py` 的具体步骤：

1. **读取数据**：`data/iris_dataset.csv`，UTF-8 编码（`utf-8-sig`，兼容带 BOM 的文件）
2. **缺失值检查**：统计并删除含缺失值的行
3. **重复行检查**：统计并删除完全重复的行
4. **类别分布**：输出每类的样本数
5. **划分数据集**：按 7:3 划为训练集/测试集，`random_state=42`
6. **Z-score 标准化**：均值和标准差**只用训练集计算**，再用同一组参数变换测试集，避免信息泄漏
7. **输出结果**：`train_scaled.csv`、`test_scaled.csv`、`scaler_params.csv`

## 实际运行结果

```
读取文件：iris_dataset.csv
数据形状：150 行 x 5 列
缺失值：0 个
重复行：1 行
  已去重，剩余 149 行
特征列：4 个 -> sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
标签列：target_name
类别分布：
setosa        50
versicolor    50
virginica     49
划分（random_state=42）：训练集 104 行，测试集 45 行
标准化：均值和标准差只用训练集计算（避免信息泄漏）
已输出：train_scaled.csv、test_scaled.csv、scaler_params.csv
```

## MNIST 数据集准备

```bash
python prepare_mnist.py
```

脚本流程：

1. **下载** 4 个 IDX 原始文件（约 11 MB）到 `data/mnist_raw/`，已存在则跳过
2. **解析**：用 `struct` 读文件头（magic、行数、28、28），用 NumPy 把字节流转成数组并展平为 784 列
3. **校验**：训练集 60000 行、测试集 10000 行；标签 0-9；像素 0-255
4. **保存**：`data/mnist_train.csv`、`data/mnist_test.csv`

输出格式：

| 项 | 说明 |
| --- | --- |
| 表头 | **无**（`header=False`） |
| 索引列 | **无**（`index=False`） |
| 列数 | 785 列（1 列标签 + 784 列像素） |
| 第 1 列 | label，取值 0-9 |
| 第 2-785 列 | 像素，取值 0-255，**未归一化**，28×28 展平 |

实测输出：

```
  图像尺寸：28 x 28，已展平为 784 列
  训练集：60000 行     测试集：10000 行
  标签范围：0-9
  像素范围：0-255（未归一化）
  mnist_train.csv  104.6 MB
  mnist_test.csv  17.5 MB
```

用 pandas 读取验证：

```python
import pandas as pd
df = pd.read_csv("data/mnist_train.csv", header=None)
print(df.shape)          # (60000, 785)
print(df[0].min(), df[0].max())          # 0 9
print(df.iloc[:, 1:].values.max())       # 255
```

## 波士顿房价数据集准备

```bash
python prepare_boston.py
```

原始数据来源：UCI `housing.data`（空白分隔、无表头、506 行 × 14 列）。

处理流程：

1. **下载** `housing.data` 到 `data/boston_raw/`（已存在则跳过）
2. **解析**：`pd.read_csv(sep=r"\s+", header=None, skipinitialspace=True)`，按一个或多个空格切分
3. **命名**：按题目给定顺序赋 14 个列名
4. **校验**：形状必须是 (506, 14)，列名完全一致，无缺失值
5. **保存**：`data/boston_housing.csv`（带表头、无索引、UTF-8）

列名顺序：

```
CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
```

实测输出：

```
  读取结果：506 行 x 14 列
  缺失值：0 个
  数值范围：最小 0.0000，最大 711.0000
  boston_housing.csv  36.8 KB
```

**说明**：全程未使用已弃用的 `sklearn.datasets.load_boston`，也未删除任何行或修改列名。

## 泰坦尼克号数据集准备

```bash
python prepare_titanic.py
```

获取 Kaggle 训练集原始内容（891 行 × 12 列），原样保存，**不做任何清洗**。

处理原则：

1. **12 列全部保留**，列名与 Kaggle 原始文件完全一致
2. **不删行**：891 行一行不删
3. **缺失值保留**：Age（177 个）、Cabin（687 个）、Embarked（2 个）保持为空字段
4. **Sex 保持文本**：`male` / `female`，不转 0/1

实测输出：

```
  读取结果：891 行 x 12 列
  列名（12 列）：PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
  缺失值统计：
    Age          177 个
    Cabin        687 个
    Embarked       2 个
  Sex 取值：['female', 'male']（保持文本，未转 0/1）
  titanic_train.csv  61.4 KB
```

来源：`https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`（Kaggle 竞赛 `train.csv` 的镜像副本）。下载失败时自动回退到本地副本。

## 西瓜数据集 3.0 准备

```bash
python prepare_watermelon.py
```

**版本明确**：周志华《机器学习》（清华大学出版社，2016）第 4 章**表 4.1「西瓜数据集 3.0」**，共 17 个样本（好瓜 8 个、坏瓜 9 个）。

字段（9 列）：`色泽、根蒂、敲声、纹理、脐部、触感、密度、含糖率、好瓜`

| 类型 | 字段 | 取值 |
| --- | --- | --- |
| 离散（中文文本） | 色泽 | 青绿、乌黑、浅白 |
| 离散（中文文本） | 根蒂 | 蜷缩、稍蜷、硬挺 |
| 离散（中文文本） | 敲声 | 浊响、沉闷、清脆 |
| 离散（中文文本） | 纹理 | 清晰、稍糊、模糊 |
| 离散（中文文本） | 脐部 | 凹陷、稍凹、平坦 |
| 离散（中文文本） | 触感 | 硬滑、软粘 |
| 连续（浮点） | 密度 | 0.243 ~ 0.774 |
| 连续（浮点） | 含糖率 | 0.042 ~ 0.460 |
| 标签 | 好瓜 | 是、否 |

**离散特征未被编码为数字**：全部保持中文文本，脚本会校验每列值域并为 `str` 类型，不符合直接报错退出。

实测输出：

```
  样本数：17 行，字段数：9 列
  好瓜分布：是 8 个，否 9 个
  离散字段取值均为中文文本，未编码为数字
  密度范围：0.243 ~ 0.774
  含糖率范围：0.042 ~ 0.460
  watermelon3.0.csv  1070 字节
```

> 如需保留原始表格的「编号」列（10 列），把脚本第 47 行 `INCLUDE_ID = False` 改成 `True`。
> 若要用**西瓜数据集 4.0**（30 个样本、多一个「含糖率」以外的字段），必须单独命名文件并说明用途，不能与本文件混用。

## CSV 规范检查

`check_csv.py` 用来验证 `data/` 下每个 CSV 是否符合统一规范：

```bash
python check_csv.py
```

检查项：

| 检查项 | 不合规的表现 |
| --- | --- |
| UTF-8 编码 | 中文乱码、非法字节 |
| 英文逗号分隔 | 用分号或制表符分隔 |
| 单行表头 | 第二行也是表头（多级表头） |
| 无多余索引列 | 首列是 `Unnamed: 0` |
| 缺失值为空字段 | 写成 `NaN`、`None` 字符串 |
| 无千分位逗号 | 数字写成 `1,234` |
| 数值不带单位 | 写成 `5.1cm` |

退出码为 0 表示全部合规，1 表示存在问题。

## 数据校验与读取测试

```bash
python validate_data.py
```

对 9 个 CSV 逐个断言：能否用 `pandas.read_csv` 读取、行数、列数、列名、缺失值分布；MNIST 额外检查标签与像素范围。

实测输出：

```
[通过] iris_dataset.csv       原始数据
       150 行 x 5 列   缺失 0
[通过] train_scaled.csv       训练集（标准化后）
       104 行 x 5 列   缺失 0
[通过] test_scaled.csv        测试集（标准化后）
       45 行 x 5 列   缺失 0
[通过] scaler_params.csv      标准化参数
       4 行 x 3 列   缺失 0
[通过] mnist_train.csv        MNIST 训练集（无表头）
       60000 行 x 785 列   缺失 0   标签 0-9   像素 0-255
[通过] mnist_test.csv         MNIST 测试集（无表头）
       10000 行 x 785 列   缺失 0   标签 0-9   像素 0-255
[通过] boston_housing.csv     波士顿房价
       506 行 x 14 列   缺失 0
[通过] titanic_train.csv      泰坦尼克号（缺失值需保留）
       891 行 x 12 列   缺失 866
[通过] watermelon3.0.csv      西瓜数据集 3.0
       17 行 x 9 列   缺失 0

结果：9/9 个数据集通过校验
```

泰坦尼克的 866 个缺失值 = Age 177 + Cabin 687 + Embarked 2，是**应当保留**的原始缺失，不是错误。

## 命名与编码规范

| 规范 | 做法 |
| --- | --- |
| 目录名 | `学号_姓名_数据准备` |
| 数据目录 | 统一放在 `data/` 下 |
| 脚本命名 | 小写 + 下划线，如 `prepare_data.py` |
| 文件名 | 不含空格，不用中文（数据文件除外） |
| 文件编码 | 统一 **UTF-8**，脚本首行声明 `# -*- coding: utf-8 -*-` |
| CSV 编码 | 读写统一用 `utf-8-sig`，Excel 打开不乱码 |

## 可复现性

划分使用固定随机种子 `SEED = 42`（脚本第 8 行），重复运行得到完全相同的训练集/测试集划分。
