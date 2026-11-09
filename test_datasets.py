# -*- coding: utf-8 -*-
"""数据集自查与单元校验。

用法：
    python test_datasets.py

全部通过退出码为 0，任一失败为 1。可直接提交前运行。
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

COLUMNS_IRIS = ["sepal length (cm)", "sepal width (cm)",
                "petal length (cm)", "petal width (cm)", "target_name"]
COLUMNS_SCALER = ["feature", "mean", "std"]
COLUMNS_BOSTON = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                  "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
COLUMNS_TITANIC = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
                   "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
COLUMNS_WATERMELON = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜"]

ALL_FILES = [
    "iris_dataset.csv", "train_scaled.csv", "test_scaled.csv", "scaler_params.csv",
    "mnist_train.csv", "mnist_test.csv", "boston_housing.csv",
    "titanic_train.csv", "watermelon3.0.csv",
]


def read(name, **kwargs):
    path = DATA_DIR / name
    assert path.is_file(), f"文件不存在：{name}"
    return pd.read_csv(path, **kwargs)


# ============================ 通用 ============================
def test_all_files_exist():
    missing = [n for n in ALL_FILES if not (DATA_DIR / n).is_file()]
    assert not missing, f"缺少文件：{missing}"


def test_all_csv_readable():
    for name in ALL_FILES:
        header = None if name.startswith("mnist") else 0
        df = read(name, header=header)
        assert df.shape[0] > 0 and df.shape[1] > 0, f"{name} 为空表"


def test_no_index_column():
    for name in ALL_FILES:
        header = None if name.startswith("mnist") else 0
        df = read(name, header=header)
        assert not any(str(c).startswith("Unnamed:") for c in df.columns), \
            f"{name} 含有索引列"


def test_utf8_encodable():
    for name in ALL_FILES:
        raw = (DATA_DIR / name).read_bytes()
        raw.decode("utf-8-sig")          # 解码失败即抛 UnicodeDecodeError


# ============================ Iris ============================
def test_iris_shape():
    df = read("iris_dataset.csv")
    assert df.shape == (150, 5), f"期望 (150, 5)，实际 {df.shape}"


def test_iris_columns():
    df = read("iris_dataset.csv")
    assert list(df.columns) == COLUMNS_IRIS, f"列名不一致：{list(df.columns)}"


def test_iris_class_balance():
    df = read("iris_dataset.csv")
    counts = df["target_name"].value_counts()
    assert set(counts.index) == {"setosa", "versicolor", "virginica"}, counts.index.tolist()
    assert counts.max() - counts.min() <= 1, "类别分布不均衡"


def test_iris_scaled_mean_zero():
    df = read("train_scaled.csv")
    means = df[COLUMNS_IRIS[:4]].mean().abs()
    assert (means < 1e-9).all(), f"训练集标准化后均值不为 0：{means.tolist()}"


def test_iris_scaled_std_one():
    df = read("train_scaled.csv")
    stds = df[COLUMNS_IRIS[:4]].std()
    assert (stds.sub(1).abs() < 1e-9).all(), f"训练集标准化后标准差不为 1：{stds.tolist()}"


def test_iris_split_sizes():
    train = read("train_scaled.csv")
    test = read("test_scaled.csv")
    assert (train.shape[0], test.shape[0]) == (104, 45), \
        f"划分应为 104/45，实际 {train.shape[0]}/{test.shape[0]}"


def test_scaler_params():
    df = read("scaler_params.csv")
    assert list(df.columns) == COLUMNS_SCALER, f"列名不一致：{list(df.columns)}"
    assert df.shape == (4, 3), f"期望 (4, 3)，实际 {df.shape}"


# ============================ MNIST ============================
def test_mnist_train_shape():
    df = read("mnist_train.csv", header=None)
    assert df.shape == (60000, 785), f"期望 (60000, 785)，实际 {df.shape}"


def test_mnist_test_shape():
    df = read("mnist_test.csv", header=None)
    assert df.shape == (10000, 785), f"期望 (10000, 785)，实际 {df.shape}"


def test_mnist_label_range():
    for name in ("mnist_train.csv", "mnist_test.csv"):
        labels = read(name, header=None).iloc[:, 0]
        assert labels.min() >= 0 and labels.max() <= 9, \
            f"{name} 标签范围异常：{labels.min()}-{labels.max()}"
        assert labels.nunique() == 10, f"{name} 标签类别不足 10 个"


def test_mnist_pixel_range():
    for name in ("mnist_train.csv", "mnist_test.csv"):
        pixels = read(name, header=None).iloc[:, 1:]
        assert pixels.values.min() >= 0 and pixels.values.max() <= 255, \
            f"{name} 像素范围异常：{pixels.values.min()}-{pixels.values.max()}"
        assert pixels.values.max() > 1, f"{name} 像素疑似被归一化到 0-1"


def test_mnist_no_header():
    df = read("mnist_train.csv", header=None)
    assert all(isinstance(c, int) for c in df.columns), "MNIST 不应有表头"


def test_mnist_no_missing():
    for name in ("mnist_train.csv", "mnist_test.csv"):
        assert int(read(name, header=None).isna().sum().sum()) == 0, f"{name} 存在缺失值"


# ============================ 波士顿房价 ============================
def test_boston_shape():
    df = read("boston_housing.csv")
    assert df.shape == (506, 14), f"期望 (506, 14)，实际 {df.shape}"


def test_boston_columns():
    df = read("boston_housing.csv")
    assert list(df.columns) == COLUMNS_BOSTON, f"列名不一致：{list(df.columns)}"


def test_boston_no_missing():
    df = read("boston_housing.csv")
    assert int(df.isna().sum().sum()) == 0, "存在缺失值"


def test_boston_all_numeric():
    df = read("boston_housing.csv")
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns), "存在非数值列"


def test_boston_value_range():
    df = read("boston_housing.csv")
    assert df["MEDV"].min() >= 5 and df["MEDV"].max() <= 51, \
        f"MEDV 范围异常：{df['MEDV'].min()}-{df['MEDV'].max()}"
    assert set(df["CHAS"].unique()) <= {0, 1}, f"CHAS 应为 0/1，实际 {df['CHAS'].unique()}"


# ============================ 泰坦尼克号 ============================
def test_titanic_shape():
    df = read("titanic_train.csv")
    assert df.shape == (891, 12), f"期望 (891, 12)，实际 {df.shape}"


def test_titanic_columns_complete():
    df = read("titanic_train.csv")
    assert list(df.columns) == COLUMNS_TITANIC, f"列名不一致：{list(df.columns)}"


def test_titanic_missing_preserved():
    df = read("titanic_train.csv")
    missing = df.isna().sum()
    assert int(missing["Age"]) == 177, f"Age 缺失应为 177，实际 {int(missing['Age'])}"
    assert int(missing["Cabin"]) == 687, f"Cabin 缺失应为 687，实际 {int(missing['Cabin'])}"
    assert int(missing["Embarked"]) == 2, f"Embarked 缺失应为 2，实际 {int(missing['Embarked'])}"


def test_titanic_sex_is_text():
    df = read("titanic_train.csv")
    values = set(df["Sex"].dropna().unique())
    assert values == {"male", "female"}, f"Sex 取值异常：{values}"


def test_titanic_target_binary():
    df = read("titanic_train.csv")
    assert set(df["Survived"].unique()) == {0, 1}, \
        f"Survived 应为 0/1，实际 {df['Survived'].unique()}"


# ============================ 西瓜数据集 3.0 ============================
def test_watermelon_shape():
    df = read("watermelon3.0.csv")
    assert df.shape == (17, 9), f"期望 (17, 9)，实际 {df.shape}"


def test_watermelon_columns_complete():
    df = read("watermelon3.0.csv")
    assert list(df.columns) == COLUMNS_WATERMELON, f"字段不一致：{list(df.columns)}"


def test_watermelon_discrete_is_text():
    df = read("watermelon3.0.csv")
    for col in ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]:
        assert df[col].map(type).eq(str).all(), f"字段「{col}」不是文本，可能被编码成数字"


def test_watermelon_discrete_domains():
    df = read("watermelon3.0.csv")
    assert set(df["色泽"]) <= {"青绿", "乌黑", "浅白"}, set(df["色泽"])
    assert set(df["根蒂"]) <= {"蜷缩", "稍蜷", "硬挺"}, set(df["根蒂"])
    assert set(df["敲声"]) <= {"浊响", "沉闷", "清脆"}, set(df["敲声"])
    assert set(df["纹理"]) <= {"清晰", "稍糊", "模糊"}, set(df["纹理"])
    assert set(df["脐部"]) <= {"凹陷", "稍凹", "平坦"}, set(df["脐部"])
    assert set(df["触感"]) <= {"硬滑", "软粘"}, set(df["触感"])


def test_watermelon_label_counts():
    df = read("watermelon3.0.csv")
    counts = df["好瓜"].value_counts()
    assert int(counts.get("是", 0)) == 8, f"好瓜应 8 个，实际 {counts.get('是', 0)}"
    assert int(counts.get("否", 0)) == 9, f"坏瓜应 9 个，实际 {counts.get('否', 0)}"


def test_watermelon_continuous_range():
    df = read("watermelon3.0.csv")
    assert df["密度"].between(0, 1).all(), "密度应在 0-1"
    assert df["含糖率"].between(0, 1).all(), "含糖率应在 0-1"


# ============================ 运行器 ============================
TESTS = [(name, obj) for name, obj in sorted(globals().items())
         if name.startswith("test_") and callable(obj)]


def main():
    if not DATA_DIR.is_dir():
        print(f"[错误] 找不到数据目录：{DATA_DIR}")
        print("请先运行 python prepare_data.py")
        return 1

    print(f"数据目录：{DATA_DIR}")
    print(f"用例数量：{len(TESTS)}\n")

    passed, failures = 0, []
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
            print(f"[通过] {name}")
        except AssertionError as exc:
            failures.append((name, str(exc)))
            print(f"[失败] {name}  -> {exc}")
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[错误] {name}  -> {type(exc).__name__}: {exc}")

    print()
    print(f"结果：{passed}/{len(TESTS)} 通过")
    if failures:
        print("\n失败明细：")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 1

    print("全部校验通过，可以提交。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
