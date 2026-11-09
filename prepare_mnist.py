# -*- coding: utf-8 -*-
"""MNIST 数据集准备：下载 IDX 文件，解析后保存为 CSV。

输出文件（data/ 目录下）：
    mnist_train.csv   60000 行 x 785 列
    mnist_test.csv    10000 行 x 785 列

CSV 格式：
    无表头、无索引列
    第 1 列    label，取值 0-9
    第 2-785 列 像素，取值 0-255（未归一化），28x28 展平
"""

import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "mnist_raw"

EXPECTED_ROWS = {"train": 60000, "test": 10000}


def download(key):
    """下载 IDX 文件，已存在则跳过。"""
    filename = FILES[key]
    path = CACHE_DIR / filename
    if path.is_file():
        print(f"  已存在，跳过下载：{filename}")
        return path
    print(f"  下载：{BASE_URL + filename}")
    urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def load_images(path):
    """解析 IDX3 图像文件，返回 (n, 784) 的 uint8 数组。"""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path.name} 不是 IDX 图像文件（magic={magic}，应为 2051）")
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols), rows, cols


def load_labels(path):
    """解析 IDX1 标签文件，返回 (n,) 的 uint8 数组。"""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path.name} 不是 IDX 标签文件（magic={magic}，应为 2049）")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def save_csv(images, labels, out_path):
    """拼接标签与像素，保存为无表头、无索引的 CSV。"""
    combined = np.column_stack([labels.reshape(-1, 1), images])
    np.savetxt(out_path, combined, fmt="%d", delimiter=",")
    return combined.shape


def check(images, labels, name):
    """校验行数、列数、取值范围。"""
    ok = True
    n, d = images.shape
    if n != EXPECTED_ROWS[name]:
        print(f"  [错误] {name} 行数应为 {EXPECTED_ROWS[name]}，实际 {n}")
        ok = False
    if d != 784:
        print(f"  [错误] {name} 像素列数应为 784，实际 {d}")
        ok = False
    if len(labels) != n:
        print(f"  [错误] {name} 标签数与样本数不一致：{len(labels)} vs {n}")
        ok = False
    if labels.min() < 0 or labels.max() > 9:
        print(f"  [错误] {name} 标签范围应在 0-9，实际 {labels.min()}-{labels.max()}")
        ok = False
    if images.min() < 0 or images.max() > 255:
        print(f"  [错误] {name} 像素范围应在 0-255，实际 {images.min()}-{images.max()}")
        ok = False
    return ok


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("1. 下载原始 IDX 文件")
    paths = {key: download(key) for key in FILES}

    print("\n2. 解析 IDX 文件")
    train_images, rows, cols = load_images(paths["train_images"])
    train_labels = load_labels(paths["train_labels"])
    test_images, _, _ = load_images(paths["test_images"])
    test_labels = load_labels(paths["test_labels"])
    print(f"  图像尺寸：{rows} x {cols}，已展平为 {rows * cols} 列")
    print(f"  训练集：{train_images.shape[0]} 行     测试集：{test_images.shape[0]} 行")

    print("\n3. 校验数据")
    ok_train = check(train_images, train_labels, "train")
    ok_test = check(test_images, test_labels, "test")
    print(f"  标签范围：{train_labels.min()}-{train_labels.max()}")
    print(f"  像素范围：{train_images.min()}-{train_images.max()}（未归一化）")
    if not (ok_train and ok_test):
        print("\n[失败] 数据校验未通过，已停止保存")
        return 1

    print("\n4. 保存 CSV（无表头、无索引）")
    train_path = DATA_DIR / "mnist_train.csv"
    test_path = DATA_DIR / "mnist_test.csv"
    save_csv(train_images, train_labels, train_path)
    save_csv(test_images, test_labels, test_path)
    print(f"  {train_path.name}  {train_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  {test_path.name}  {test_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
