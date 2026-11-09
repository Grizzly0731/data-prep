# -*- coding: utf-8 -*-
"""西瓜数据集 3.0 准备：按周志华《机器学习》表 4.1 录入，保存为 CSV。

版本说明：
    周志华《机器学习》（清华大学出版社，2016）第 4 章表 4.1「西瓜数据集 3.0」，
    共 17 个样本（好瓜 8 个、坏瓜 9 个），6 个离散特征 + 2 个连续特征。

输出：data/watermelon3.0.csv
    字段：色泽、根蒂、敲声、纹理、脐部、触感、密度、含糖率、好瓜
    离散特征保持中文文本，未编码为数字
"""

import sys
from pathlib import Path

import pandas as pd

# 原始表格（编号, 色泽, 根蒂, 敲声, 纹理, 脐部, 触感, 密度, 含糖率, 好瓜）
ROWS = [
    (1,  "青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460, "是"),
    (2,  "乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.774, 0.376, "是"),
    (3,  "乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.634, 0.264, "是"),
    (4,  "青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.608, 0.318, "是"),
    (5,  "浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.556, 0.215, "是"),
    (6,  "青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.403, 0.237, "是"),
    (7,  "乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 0.481, 0.149, "是"),
    (8,  "乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 0.437, 0.211, "是"),
    (9,  "乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0.666, 0.091, "否"),
    (10, "青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0.243, 0.267, "否"),
    (11, "浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0.245, 0.057, "否"),
    (12, "浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0.343, 0.099, "否"),
    (13, "青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0.639, 0.161, "否"),
    (14, "浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0.657, 0.198, "否"),
    (15, "乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.360, 0.370, "否"),
    (16, "浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0.593, 0.042, "否"),
    (17, "青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0.719, 0.103, "否"),
]

ALL_COLUMNS = ["编号", "色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜"]
FIELDS = ALL_COLUMNS[1:]          # 题目要求的 9 个字段
INCLUDE_ID = False                # 改成 True 可保留原始「编号」列
DISCRETE = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]
DOMAINS = {
    "色泽": {"青绿", "乌黑", "浅白"},
    "根蒂": {"蜷缩", "稍蜷", "硬挺"},
    "敲声": {"浊响", "沉闷", "清脆"},
    "纹理": {"清晰", "稍糊", "模糊"},
    "脐部": {"凹陷", "稍凹", "平坦"},
    "触感": {"硬滑", "软粘"},
    "好瓜": {"是", "否"},
}
EXPECTED_ROWS = 17

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_FILE = DATA_DIR / "watermelon3.0.csv"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("1. 构建数据表（版本：西瓜数据集 3.0，周志华《机器学习》表 4.1）")
    columns = ALL_COLUMNS if INCLUDE_ID else FIELDS
    keep = list(range(len(ALL_COLUMNS))) if INCLUDE_ID else list(range(1, len(ALL_COLUMNS)))
    df = pd.DataFrame([[row[i] for i in keep] for row in ROWS], columns=columns)
    print(f"  样本数：{df.shape[0]} 行，字段数：{df.shape[1]} 列")

    print("\n2. 校验")
    ok = True
    if df.shape[0] != EXPECTED_ROWS:
        print(f"  [错误] 行数应为 {EXPECTED_ROWS}，实际 {df.shape[0]}")
        ok = False
    expected_cols = ALL_COLUMNS if INCLUDE_ID else FIELDS
    if list(df.columns) != expected_cols:
        print(f"  [错误] 字段与要求不一致：{list(df.columns)}")
        ok = False
    if int(df.isna().sum().sum()) != 0:
        print("  [错误] 存在缺失值")
        ok = False
    for col, domain in DOMAINS.items():
        actual = set(df[col].unique())
        if not actual.issubset(domain):
            print(f"  [错误] 字段「{col}」出现非预期取值：{actual - domain}")
            ok = False
        if df[col].map(type).ne(str).any():
            print(f"  [错误] 字段「{col}」不是文本类型，可能被编码成了数字")
            ok = False
    for col in ("密度", "含糖率"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"  [错误] 字段「{col}」不是数值类型")
            ok = False
    if not ok:
        print("\n[失败] 校验未通过，已停止保存")
        return 1

    print(f"  字段：{', '.join(df.columns)}")
    print(f"  好瓜分布：是 {int((df['好瓜'] == '是').sum())} 个，否 {int((df['好瓜'] == '否').sum())} 个")
    print(f"  离散字段取值均为中文文本，未编码为数字")
    print(f"  密度范围：{df['密度'].min():.3f} ~ {df['密度'].max():.3f}")
    print(f"  含糖率范围：{df['含糖率'].min():.3f} ~ {df['含糖率'].max():.3f}")

    print("\n3. 保存 CSV（UTF-8、带表头、无索引）")
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"  {OUT_FILE.name}  {OUT_FILE.stat().st_size} 字节")

    check = pd.read_csv(OUT_FILE)
    print(f"  回读校验：{check.shape[0]} 行 x {check.shape[1]} 列")
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
