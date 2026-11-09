# -*- coding: utf-8 -*-
"""数据校验与读取测试。

对 data/ 下每个 CSV 做「读取 + 断言」检查：

    - 能否用 pandas.read_csv 成功读取
    - 行数、列数、列名是否与预期一致
    - 缺失值分布是否与预期一致
    - MNIST 额外检查标签范围（0-9）与像素范围（0-255）

全部通过时退出码为 0，任一失败为 1。

用法：
    python validate_data.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

COLUMNS_IRIS = ["sepal length (cm)", "sepal width (cm)",
                "petal length (cm)", "petal width (cm)", "target_name"]
COLUMNS_BOSTON = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                  "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
COLUMNS_TITANIC = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
                   "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
COLUMNS_WATERMELON = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜"]

SPECS = [
    {
        "file": "iris_dataset.csv", "rows": 150, "cols": 5,
        "columns": COLUMNS_IRIS, "missing": {},
        "note": "原始数据",
    },
    {
        "file": "train_scaled.csv", "rows": 104, "cols": 5,
        "columns": COLUMNS_IRIS, "missing": {},
        "note": "训练集（标准化后）",
    },
    {
        "file": "test_scaled.csv", "rows": 45, "cols": 5,
        "columns": COLUMNS_IRIS, "missing": {},
        "note": "测试集（标准化后）",
    },
    {
        "file": "scaler_params.csv", "rows": 4, "cols": 3,
        "columns": ["feature", "mean", "std"], "missing": {},
        "note": "标准化参数",
    },
    {
        "file": "mnist_train.csv", "rows": 60000, "cols": 785,
        "columns": None, "missing": {}, "header": None,
        "label_range": (0, 9), "pixel_range": (0, 255),
        "note": "MNIST 训练集（无表头）",
    },
    {
        "file": "mnist_test.csv", "rows": 10000, "cols": 785,
        "columns": None, "missing": {}, "header": None,
        "label_range": (0, 9), "pixel_range": (0, 255),
        "note": "MNIST 测试集（无表头）",
    },
    {
        "file": "boston_housing.csv", "rows": 506, "cols": 14,
        "columns": COLUMNS_BOSTON, "missing": {},
        "note": "波士顿房价",
    },
    {
        "file": "titanic_train.csv", "rows": 891, "cols": 12,
        "columns": COLUMNS_TITANIC, "missing": {"Age": 177, "Cabin": 687, "Embarked": 2},
        "note": "泰坦尼克号（缺失值需保留）",
    },
    {
        "file": "watermelon3.0.csv", "rows": 17, "cols": 9,
        "columns": COLUMNS_WATERMELON, "missing": {},
        "note": "西瓜数据集 3.0",
    },
]


def check_one(spec):
    """校验单个文件，返回 (是否通过, 问题列表, 摘要信息)。"""
    path = DATA_DIR / spec["file"]
    problems = []

    if not path.is_file():
        return False, [f"文件不存在：{path}"], ""

    # 1. 能否用 pandas 读取
    try:
        df = pd.read_csv(path, header=spec.get("header", 0))
    except Exception as exc:
        return False, [f"pandas.read_csv 失败：{type(exc).__name__}: {exc}"], ""

    # 2. 行列数
    if df.shape[0] != spec["rows"]:
        problems.append(f"行数应为 {spec['rows']}，实际 {df.shape[0]}")
    if df.shape[1] != spec["cols"]:
        problems.append(f"列数应为 {spec['cols']}，实际 {df.shape[1]}")

    # 3. 列名
    if spec["columns"] is not None:
        actual = list(df.columns)
        if actual != spec["columns"]:
            problems.append(f"列名不一致：{actual}")

    # 4. 缺失值分布
    missing = df.isna().sum()
    expected_missing = spec["missing"]
    for col in df.columns:
        actual_n = int(missing[col])
        expected_n = expected_missing.get(col, 0)
        if actual_n != expected_n:
            problems.append(f"列「{col}」缺失值应为 {expected_n}，实际 {actual_n}")

    # 5. MNIST 标签与像素范围
    if spec.get("header", 0) is None:
        labels = df.iloc[:, 0]
        pixels = df.iloc[:, 1:]
        low, high = spec["label_range"]
        if labels.min() < low or labels.max() > high:
            problems.append(f"标签范围应在 {low}-{high}，实际 {labels.min()}-{labels.max()}")
        plow, phigh = spec["pixel_range"]
        if pixels.values.min() < plow or pixels.values.max() > phigh:
            problems.append(f"像素范围应在 {plow}-{phigh}，实际 {pixels.values.min()}-{pixels.values.max()}")
        if pixels.values.dtype.kind not in "iu f":
            pass
        summary = (f"{df.shape[0]} 行 x {df.shape[1]} 列   缺失 {int(missing.sum())}   "
                   f"标签 {labels.min()}-{labels.max()}   像素 {pixels.values.min()}-{pixels.values.max()}")
        return not problems, problems, summary

    summary = f"{df.shape[0]} 行 x {df.shape[1]} 列   缺失 {int(missing.sum())}"
    return not problems, problems, summary


def main():
    if not DATA_DIR.is_dir():
        print(f"[错误] 找不到数据目录：{DATA_DIR}")
        return 1

    print(f"校验目录：{DATA_DIR}")
    print(f"文件数量：{len(SPECS)}\n")

    passed, failed = 0, 0
    for spec in SPECS:
        ok, problems, summary = check_one(spec)
        if ok:
            passed += 1
            print(f"[通过] {spec['file']:<22s} {spec['note']}")
            print(f"       {summary}")
        else:
            failed += 1
            print(f"[失败] {spec['file']:<22s} {spec['note']}")
            for item in problems:
                print(f"       - {item}")

    print()
    print(f"结果：{passed}/{len(SPECS)} 个数据集通过校验")
    if failed:
        print("提示：先运行 python prepare_data.py 生成数据文件，再运行本脚本。")
        return 1
    print("所有数据集均可正常读取，行列数、列名与缺失值符合预期。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
