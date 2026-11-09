# -*- coding: utf-8 -*-
"""Iris 数据准备脚本。

原始数据：data/iris_dataset.csv（课程提供的原始表，150 行 x 5 列，UTF-8 编码，自带表头）
转换方式：pandas 读取 -> 查缺失/重复 -> 7:3 划分 -> 用训练集统计量做 Z-score
输出文件：data/train_scaled.csv、data/test_scaled.csv、data/scaler_params.csv
"""

import sys
from pathlib import Path

import pandas as pd

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "iris_dataset.csv"


def main():
    if not RAW_CSV.is_file():
        print(f"[错误] 找不到数据文件：{RAW_CSV}")
        return 1

    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    print(f"读取文件：{RAW_CSV.name}")
    print(f"数据形状：{df.shape[0]} 行 x {df.shape[1]} 列")

    n_missing = int(df.isna().sum().sum())
    print(f"缺失值：{n_missing} 个")
    if n_missing:
        df = df.dropna().reset_index(drop=True)
        print(f"  已删除含缺失值的行，剩余 {len(df)} 行")

    n_dup = int(df.duplicated().sum())
    print(f"重复行：{n_dup} 行")
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"  已去重，剩余 {len(df)} 行")

    label_col = df.columns[-1]
    feature_cols = list(df.columns[:-1])
    print(f"特征列：{len(feature_cols)} 个 -> {', '.join(feature_cols)}")
    print(f"标签列：{label_col}")
    print("类别分布：")
    print(df[label_col].value_counts().to_string())

    train = df.sample(frac=0.7, random_state=SEED)
    test = df.drop(train.index)
    print(f"划分（random_state={SEED}）：训练集 {len(train)} 行，测试集 {len(test)} 行")

    mean = train[feature_cols].mean()
    std = train[feature_cols].std()
    train_out = train.copy()
    test_out = test.copy()
    train_out[feature_cols] = (train[feature_cols] - mean) / std
    test_out[feature_cols] = (test[feature_cols] - mean) / std
    print("标准化：均值和标准差只用训练集计算（避免信息泄漏）")

    train_path = DATA_DIR / "train_scaled.csv"
    test_path = DATA_DIR / "test_scaled.csv"
    param_path = DATA_DIR / "scaler_params.csv"
    train_out.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_out.to_csv(test_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols,
                  "mean": mean.to_numpy(),
                  "std": std.to_numpy()}).to_csv(param_path, index=False, encoding="utf-8-sig")
    print(f"已输出：{train_path.name}、{test_path.name}、{param_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
