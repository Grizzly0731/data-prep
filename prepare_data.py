# -*- coding: utf-8 -*-
"""数据获取与转换总入口。

一条命令完成「原始数据 -> CSV」的全部转换，并说明每份数据：

    - 从哪来（原始文件或下载地址）
    - 用什么代码转换（对应脚本）
    - 输出什么（行列数 + SHA-256，用于验证重复运行结果一致）

用法：
    python prepare_data.py              # 转换全部数据集
    python prepare_data.py iris titanic # 只转换指定的几个
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PYTHON = sys.executable

DATASETS = [
    {
        "key": "iris",
        "title": "Iris 鸢尾花",
        "script": "prepare_iris.py",
        "source": "data/iris_dataset.csv（课程提供原始表，150 行 x 5 列，UTF-8）",
        "method": "pandas 读取 -> 查缺失/重复 -> 7:3 划分 -> 用训练集统计量做 Z-score",
        "outputs": ["train_scaled.csv", "test_scaled.csv", "scaler_params.csv"],
    },
    {
        "key": "mnist",
        "title": "MNIST 手写数字",
        "script": "prepare_mnist.py",
        "source": "官方 IDX 文件 https://storage.googleapis.com/cvdf-datasets/mnist/（4 个 .gz，约 11 MB）",
        "method": "gzip 解压 -> struct 解析文件头（magic/行数/28/28）-> NumPy 展平 784 维 -> 无表头保存",
        "outputs": ["mnist_train.csv", "mnist_test.csv"],
    },
    {
        "key": "boston",
        "title": "波士顿房价",
        "script": "prepare_boston.py",
        "source": "UCI https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        "method": "空白分隔文本 -> sep=r'\\\\s+' 解析 -> 按题目顺序补 14 个列名 -> 带表头保存",
        "outputs": ["boston_housing.csv"],
    },
    {
        "key": "titanic",
        "title": "泰坦尼克号",
        "script": "prepare_titanic.py",
        "source": "Kaggle train.csv 镜像 https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "method": "pandas 读取 -> 校验 891x12 与列名 -> 原样保存（缺失值、Sex 文本均不改动）",
        "outputs": ["titanic_train.csv"],
    },
    {
        "key": "watermelon",
        "title": "西瓜数据集 3.0",
        "script": "prepare_watermelon.py",
        "source": "周志华《机器学习》表 4.1（17 个样本，内嵌在脚本中，无需下载）",
        "method": "DataFrame 录入 -> 校验字段与值域 -> UTF-8-SIG 保存（离散特征保持中文）",
        "outputs": ["watermelon3.0.csv"],
    },
]


def decode(raw):
    """脚本输出可能是 UTF-8 或 GBK，依次尝试解码。"""
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def sha256_of(path, chunk=1 << 20):
    """返回文件 SHA-256 前 16 位，用于比对两次运行是否一致。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def print_registry(datasets):
    print("=" * 72)
    print("数据来源登记表")
    print("=" * 72)
    for i, item in enumerate(datasets, 1):
        print(f"[{i}] {item['title']}（{item['key']}）")
        print(f"    来源：{item['source']}")
        print(f"    转换：{item['method']}")
        print(f"    脚本：{item['script']}")
        print(f"    输出：{', '.join(item['outputs'])}")
    print()


def run_one(item):
    script = BASE_DIR / item["script"]
    print("-" * 72)
    print(f">> {item['title']}  ->  {item['script']}")
    print("-" * 72)
    if not script.is_file():
        print(f"  [错误] 找不到脚本 {script.name}")
        return False

    proc = subprocess.run([PYTHON, str(script)], cwd=str(BASE_DIR), capture_output=True)
    output = decode(proc.stdout).strip()
    if output:
        for line in output.splitlines():
            print(f"  {line}")
    if proc.returncode != 0:
        err = decode(proc.stderr).strip()
        print(f"  [失败] 退出码 {proc.returncode}")
        if err:
            for line in err.splitlines()[-5:]:
                print(f"    {line}")
        return False
    return True


def summarize(datasets):
    print()
    print("=" * 72)
    print("输出文件校验（两次运行 SHA-256 相同即可复现）")
    print("=" * 72)
    print(f"{'文件':<26s}{'行 x 列':>14s}{'大小':>12s}  SHA-256")
    missing = 0
    for item in datasets:
        for name in item["outputs"]:
            path = DATA_DIR / name
            if not path.is_file():
                print(f"{name:<26s}{'缺失':>14s}")
                missing += 1
                continue
            rows, cols = 0, 0
            try:
                header = None if name.startswith("mnist") else 0
                df = pd.read_csv(path, header=header)
                rows, cols = df.shape
            except Exception:
                pass
            size_bytes = path.stat().st_size
            size = f"{size_bytes / 1024 / 1024:.2f} MB" if size_bytes > (1 << 20) \
                else f"{size_bytes / 1024:.1f} KB"
            print(f"{name:<26s}{f'{rows} x {cols}':>14s}{size:>12s}  {sha256_of(path)}")
    return missing


def main(argv):
    wanted = [a for a in argv[1:] if not a.startswith("-")]
    datasets = [d for d in DATASETS if not wanted or d["key"] in wanted]
    if not datasets:
        print(f"没有匹配的数据集：{wanted}")
        print(f"可选：{', '.join(d['key'] for d in DATASETS)}")
        return 1

    print(f"Python：{PYTHON}")
    print(f"工作目录：{BASE_DIR}")
    print(f"待转换：{len(datasets)} 个数据集\n")

    print_registry(datasets)

    failed = []
    for item in datasets:
        if not run_one(item):
            failed.append(item["key"])
        print()

    missing = summarize(datasets)

    print()
    if failed or missing:
        print(f"[完成但有问题] 失败脚本：{failed or '无'}；缺失输出：{missing} 个")
        return 1
    print("[全部完成] 所有数据集已转换成功。")
    print("重复运行本脚本，上表 SHA-256 应完全一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
