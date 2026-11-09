# -*- coding: utf-8 -*-
"""波士顿房价数据集准备：从 UCI 原始 housing.data 转换为 CSV。

原始文件特点：
    - 无表头
    - 每行 14 个数值，用空格分隔（部分行首有空格）
    - 506 行

输出：data/boston_housing.csv
    506 行 x 14 列，带表头，无索引列
"""

import sys
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
COLUMNS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
           "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
EXPECTED_SHAPE = (506, 14)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "boston_raw"
RAW_FILE = RAW_DIR / "housing.data"
OUT_FILE = DATA_DIR / "boston_housing.csv"


def download():
    """下载原始 housing.data，已存在则跳过。"""
    if RAW_FILE.is_file():
        print(f"  已存在，跳过下载：{RAW_FILE.name}")
        return RAW_FILE
    print(f"  下载：{URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    RAW_FILE.write_bytes(data)
    print(f"  已保存原始文件：{RAW_FILE.name}（{len(data)} 字节）")
    return RAW_FILE


def parse(path):
    """按空白分隔读取，无表头，过滤空行。"""
    df = pd.read_csv(
        path,
        sep=r"\s+",          # 一个或多个空格 / 制表符
        header=None,         # 原始文件没有表头
        skipinitialspace=True,
        engine="python",
    )
    df = df.dropna(how="all").reset_index(drop=True)   # 去掉末尾空行
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"列数为 {df.shape[1]}，应为 {len(COLUMNS)}")
    df.columns = COLUMNS          # 按题目给定的顺序命名，不改名、不删列
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("1. 获取原始数据")
    download()

    print("\n2. 解析空白分隔文本")
    df = parse(RAW_FILE)
    print(f"  读取结果：{df.shape[0]} 行 x {df.shape[1]} 列")

    print("\n3. 校验")
    ok = True
    if df.shape != EXPECTED_SHAPE:
        print(f"  [错误] 形状应为 {EXPECTED_SHAPE}，实际 {df.shape}")
        ok = False
    if list(df.columns) != COLUMNS:
        print(f"  [错误] 列名与要求不一致：{list(df.columns)}")
        ok = False
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        print(f"  [错误] 存在 {n_missing} 个缺失值")
        ok = False
    if not ok:
        print("\n[失败] 校验未通过，已停止保存")
        return 1
    print(f"  列名：{', '.join(df.columns)}")
    print(f"  缺失值：{n_missing} 个")
    print(f"  数值范围：最小 {df.values.min():.4f}，最大 {df.values.max():.4f}")

    print("\n4. 保存 CSV（带表头、无索引）")
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"  {OUT_FILE.name}  {OUT_FILE.stat().st_size / 1024:.1f} KB")
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
