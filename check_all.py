# -*- coding: utf-8 -*-
"""异常情况演示 + 批量检查。

用法：
    python check_all.py          # 逐个检查 data/ 下的 CSV，单个失败不中断
    python check_all.py --demo   # 演示 5 类异常的处理提示
"""

import sys
import tempfile
from pathlib import Path

from safe_utils import check_columns, safe_download, safe_read_csv

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

# 文件名 -> (期望列名, 是否无表头)
TARGETS = [
    ("iris_dataset.csv", COLUMNS_IRIS, False),
    ("train_scaled.csv", COLUMNS_IRIS, False),
    ("test_scaled.csv", COLUMNS_IRIS, False),
    ("scaler_params.csv", COLUMNS_SCALER, False),
    ("mnist_train.csv", None, True),
    ("mnist_test.csv", None, True),
    ("boston_housing.csv", COLUMNS_BOSTON, False),
    ("titanic_train.csv", COLUMNS_TITANIC, False),
    ("watermelon3.0.csv", COLUMNS_WATERMELON, False),
]


def demo():
    """逐条演示 5 类异常，证明提示清晰且不抛 traceback。"""
    print("=" * 68)
    print("异常情况演示")
    print("=" * 68)

    print("\n[1] 文件不存在")
    df, err = safe_read_csv(DATA_DIR / "不存在的文件.csv")
    print(f"    结果：{err}")

    print("\n[2] 编码错误（用 ascii 强制解码中文文件）")
    df, err = safe_read_csv(DATA_DIR / "watermelon3.0.csv", encodings=("ascii",))
    print(f"    结果：{err}")

    print("\n[3] 空数据")
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.csv"
        empty.write_text("", encoding="utf-8")
        df, err = safe_read_csv(empty)
        print(f"    结果：{err}")

        blank = Path(tmp) / "blank.csv"
        blank.write_text("\n\n\n", encoding="utf-8")
        df, err = safe_read_csv(blank)
        print(f"    结果：{err}")

    print("\n[4] 列数不一致")
    df, err = safe_read_csv(DATA_DIR / "boston_housing.csv")
    if err is None:
        ok, msg = check_columns(df, COLUMNS_TITANIC, "boston_housing.csv")
        print(f"    结果：{msg}")

        only3 = df.iloc[:, :3]
        ok, msg = check_columns(only3, COLUMNS_BOSTON, "只取前 3 列")
        print(f"    结果：{msg}")

    print("\n[5] 下载失败（故意用一个不存在的域名）")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "x.gz"
        ok, msg = safe_download("https://not-exist.invalid/x.gz", dest,
                                timeout=3, retries=1)
        print(f"    结果：{'成功' if ok else '失败'}")
        for line in msg.splitlines():
            print(f"    {line}")

    print("\n[6] 下载失败但有本地副本（回退方案）")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "y.csv"
        ok, msg = safe_download("https://not-exist.invalid/y.csv", dest, timeout=3,
                                retries=1, fallbacks=[DATA_DIR / "watermelon3.0.csv"])
        print(f"    结果：{'成功' if ok else '失败'} -> {msg}")
    return 0


def check_all():
    """逐个检查，任何一个失败都只记录、不中断。"""
    print("=" * 68)
    print("批量检查（单个文件失败不影响其他文件）")
    print("=" * 68)

    if not DATA_DIR.is_dir():
        print(f"[错误] 数据目录不存在：{DATA_DIR}")
        print("请先运行 python prepare_data.py 生成数据文件。")
        return 1

    passed, failed = 0, 0
    problems = []

    for name, expected_columns, no_header in TARGETS:
        path = DATA_DIR / name
        header = None if no_header else 0
        df, err = safe_read_csv(path, header=header)

        if err is not None:
            failed += 1
            problems.append((name, err))
            print(f"[失败] {name:<22s} {err}")
            continue

        ok, msg = check_columns(df, expected_columns, name)
        if not ok:
            failed += 1
            problems.append((name, msg))
            print(f"[失败] {name:<22s} {msg}")
            continue

        passed += 1
        print(f"[通过] {name:<22s} {df.shape[0]} 行 x {df.shape[1]} 列   "
              f"缺失 {int(df.isna().sum().sum())}")

    print()
    print(f"结果：{passed} 个通过，{failed} 个失败")

    if problems:
        print("\n失败明细（脚本未中断，其余文件已正常检查）：")
        for name, msg in problems:
            print(f"  - {name}：{msg}")
        return 1

    print("全部文件均可正常读取，列名与列数符合预期。")
    return 0


def main():
    if "--demo" in sys.argv:
        return demo()
    return check_all()


if __name__ == "__main__":
    sys.exit(main())
