# -*- coding: utf-8 -*-
"""泰坦尼克号数据集准备：获取 Kaggle 训练集原始内容，另存为 CSV。

要求：
    - 保留全部 12 列，不改名、不删列、不删行
    - Age / Cabin / Embarked 的原始缺失值保留（写成空字段）
    - Sex 保持文本（male / female），不转 0/1

输出：data/titanic_train.csv
"""

import sys
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
COLUMNS = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
           "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
EXPECTED_SHAPE = (891, 12)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "titanic_raw"
RAW_FILE = RAW_DIR / "titanic.csv"
OUT_FILE = DATA_DIR / "titanic_train.csv"

# 下载失败时依次尝试的本地副本
LOCAL_FALLBACKS = [
    BASE_DIR.parent / "第3周上机实践" / "03-pandas_exercise" / "titanic.csv",
]


def fetch():
    """优先用本地缓存，其次下载，最后尝试本地副本。"""
    if RAW_FILE.is_file():
        print(f"  已存在，跳过下载：{RAW_FILE.name}")
        return RAW_FILE

    print(f"  下载：{URL}")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        RAW_FILE.write_bytes(data)
        print(f"  已保存：{RAW_FILE.name}（{len(data)} 字节）")
        return RAW_FILE
    except Exception as exc:
        print(f"  下载失败（{type(exc).__name__}），尝试本地副本")
        for path in LOCAL_FALLBACKS:
            if path.is_file():
                RAW_FILE.write_bytes(path.read_bytes())
                print(f"  已复制本地文件：{path}")
                return RAW_FILE
        raise FileNotFoundError("下载失败，且找不到本地副本，请手动把 titanic.csv 放到 data/titanic_raw/")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("1. 获取原始数据")
    fetch()

    print("\n2. 读取原始 CSV")
    df = pd.read_csv(RAW_FILE)
    print(f"  读取结果：{df.shape[0]} 行 x {df.shape[1]} 列")

    print("\n3. 校验原始内容是否完整")
    ok = True
    if df.shape != EXPECTED_SHAPE:
        print(f"  [错误] 形状应为 {EXPECTED_SHAPE}，实际 {df.shape}")
        ok = False
    if list(df.columns) != COLUMNS:
        print(f"  [错误] 列名与 Kaggle 原始文件不一致：{list(df.columns)}")
        ok = False
    if "Sex" in df.columns:
        sex_values = set(df["Sex"].dropna().unique())
        if not sex_values.issubset({"male", "female"}):
            print(f"  [错误] Sex 列取值异常：{sex_values}，可能已被转成 0/1")
            ok = False
    if not ok:
        print("\n[失败] 校验未通过，已停止保存")
        return 1

    print(f"  列名（12 列）：{', '.join(df.columns)}")
    print("  缺失值统计：")
    missing = df.isna().sum()
    for col in COLUMNS:
        if missing[col] > 0:
            print(f"    {col:<12s} {int(missing[col]):>3d} 个")
    print(f"  Sex 取值：{sorted(df['Sex'].dropna().unique().tolist())}（保持文本，未转 0/1）")

    print("\n4. 保存 CSV（保留原始缺失值，无索引）")
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig", na_rep="")
    print(f"  {OUT_FILE.name}  {OUT_FILE.stat().st_size / 1024:.1f} KB")

    check = pd.read_csv(OUT_FILE)
    print(f"  回读校验：{check.shape[0]} 行 x {check.shape[1]} 列，"
          f"Age 缺失 {int(check['Age'].isna().sum())} 个，"
          f"Cabin 缺失 {int(check['Cabin'].isna().sum())} 个")
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
