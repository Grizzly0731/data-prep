# -*- coding: utf-8 -*-
"""CSV 规范检查脚本：验证 data/ 下每个 CSV 是否符合统一规范。

规范要求：
1. UTF-8 编码（允许带 BOM）
2. 英文逗号分隔
3. 第一行是表头，且只有一行表头
4. 不保存行索引（没有 Unnamed: 0 这类列）
5. 缺失值保留为空字段（不写成 NaN / None 字符串）
6. 数字不带千分位逗号、不带单位
7. 不改列名（列名保持原始写法）
"""

import csv
import io
import re
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_COL_PATTERN = re.compile(r"^Unnamed:\s*0$|^$")
THOUSAND_PATTERN = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
UNIT_PATTERN = re.compile(r"^-?\d+(\.\d+)?\s*[a-zA-Z%]+$")
NA_STRINGS = {"NaN", "nan", "None", "none", "null", "NULL", "NA", "N/A"}


def is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def inspect(path):
    """检查单个文件，返回问题列表（空列表表示全部合规）。"""
    issues = []

    raw = path.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        encoding, payload = "utf-8-sig", raw[3:]
    else:
        encoding, payload = "utf-8", raw
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return ["编码不是 UTF-8，存在非法字节（中文可能乱码）"], None

    lines = text.splitlines()
    if not lines:
        return ["文件为空"], None

    header = lines[0]
    if header.count("\t") > 0:
        issues.append("分隔符不是英文逗号（检测到制表符）")
    elif header.count(";") > header.count(","):
        issues.append("分隔符不是英文逗号（检测到分号）")

    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)

    for col in df.columns:
        if INDEX_COL_PATTERN.match(str(col)):
            issues.append(f"存在多余索引列「{col}」，保存时应加 index=False")
            break

    if df.columns.duplicated().any():
        issues.append("表头存在重复列名")

    if len(lines) > 1:
        cells = next(csv.reader(io.StringIO(lines[1])))
        if len(cells) == len(df.columns) and not any(is_number(c) for c in cells):
            issues.append("疑似多级表头：第二行也像表头，真正的数据从第三行才开始")

    for col in df.columns:
        values = df[col]
        if values.isin(NA_STRINGS).any():
            issues.append(f"列「{col}」缺失值被写成了 NaN/None 字符串，应为空字段")
        if values.str.match(THOUSAND_PATTERN).any():
            issues.append(f"列「{col}」存在千分位逗号（如 1,234）")
        if values.str.match(UNIT_PATTERN).any():
            issues.append(f"列「{col}」数值后带单位（如 5.1cm）")

    return issues, df


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[错误] {DATA_DIR} 下没有找到 CSV 文件")
        return 1

    print(f"检查目录：{DATA_DIR}")
    print(f"文件数量：{len(csv_files)}\n")

    total_failed = 0
    for path in csv_files:
        issues, df = inspect(path)
        if issues:
            total_failed += 1
            print(f"[不合规] {path.name}")
            for item in issues:
                print(f"    - {item}")
        else:
            shape = f"{df.shape[0]} 行 x {df.shape[1]} 列" if df is not None else ""
            print(f"[合规]   {path.name}   {shape}")

    print(f"\n结果：{len(csv_files) - total_failed}/{len(csv_files)} 个文件符合规范")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
