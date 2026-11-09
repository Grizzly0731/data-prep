# -*- coding: utf-8 -*-
import random

import numpy as np
import pandas as pd

from knn import (
    KNNClassifier,
    KNNRegressor,
    accuracy_score,
    classification_report,
    confusion_matrix,
    euclidean_distance,
    find_nearest_neighbors,
    manhattan_distance,
)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


set_seed()

# ============ 1. 读取 CSV ============
df = pd.read_csv("data/iris_dataset.csv")

print("样本数（行）:", len(df))
print("特征数（列）:", len(df.columns) - 1)
print("缺失值总数:", df.isna().sum().sum())

counts = df["target_name"].value_counts()
print("\n类别分布：")
print(counts)

print("\n类别分布图：")
for name, n in counts.items():
    print(f"{name:<12s} {'#' * n} {n}")

# ============ 2. 划分 + 标准化 ============
train_df = df.sample(frac=0.7, random_state=SEED)
test_df = df.drop(train_df.index)

feature_cols = list(df.columns[:-1])
label_col = df.columns[-1]

mean = train_df[feature_cols].mean()
std = train_df[feature_cols].std()
train_scaled = (train_df[feature_cols] - mean) / std
test_scaled = (test_df[feature_cols] - mean) / std

print("\n训练集:", len(train_df), " 测试集:", len(test_df))
print("训练集标准化后均值:", train_scaled.mean().round(3).tolist())
print("训练集标准化后标准差:", train_scaled.std().round(3).tolist())
print("测试集标准化后均值:", test_scaled.mean().round(3).tolist())

# ============ 3. 距离函数单元测试 ============
a, b = [0, 0], [3, 4]
print("\n单样本 欧氏 =", euclidean_distance(a, b), " 曼哈顿 =", manhattan_distance(a, b))
assert euclidean_distance(a, b) == 5.0
assert manhattan_distance(a, b) == 7.0

A = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
B = np.array([[3, 4], [0, 0]], dtype=float)
d_e = euclidean_distance(A, B)
d_m = manhattan_distance(A, B)
assert np.allclose(d_e, np.linalg.norm(A[:, None] - B[None], axis=-1))
assert np.allclose(d_m, np.abs(A[:, None] - B[None]).sum(axis=-1))
print("欧氏矩阵:\n", d_e.round(3))
print("曼哈顿矩阵:\n", d_m.round(3))
print("单元测试全部通过")

# ============ 4. 最近邻搜索 ============
X_train_np = train_scaled.to_numpy()
X_test_np = test_scaled.to_numpy()
y_train_np = train_df[label_col].to_numpy()
y_test_np = test_df[label_col].to_numpy()

x = X_test_np[0]
idx, dist = find_nearest_neighbors(X_train_np, x, k=5)

print("\n测试样本 1 的 5 个最近邻：")
print(f"  {'排名':<4}{'训练集索引':>10}{'距离':>10}  类别")
for rank, (i, d) in enumerate(zip(idx, dist), 1):
    print(f"  {rank:<4}{i:>10d}{d:>10.4f}  {y_train_np[i]}")

# ============ 5. KNN 分类 ============
clf = KNNClassifier(k=5).fit(X_train_np, y_train_np)
y_pred = clf.predict(X_test_np)

print("\nKNN 分类结果（K = 5）：")
print("预测前 10 个:", y_pred[:10])
print("真实前 10 个:", y_test_np[:10])
print(f"测试集准确率 = {clf.score(X_test_np, y_test_np):.4f}")

toy = KNNClassifier(k=4).fit([[0], [1], [2], [3]], ["B", "A", "A", "B"])
print("平票演示（2 : 2）→", toy.predict([1.5])[0])

# ============ 6. KNN 回归 ============
Xr_raw = train_df[feature_cols[:3]].to_numpy()
yr = train_df[feature_cols[3]].to_numpy()
Xr_test_raw = test_df[feature_cols[:3]].to_numpy()
yr_test = test_df[feature_cols[3]].to_numpy()

mu = Xr_raw.mean(axis=0)
sd = Xr_raw.std(axis=0)
Xr = (Xr_raw - mu) / sd
Xr_test = (Xr_test_raw - mu) / sd

print("\nKNN 回归（预测花瓣宽度）：")
for w in ("uniform", "distance"):
    reg = KNNRegressor(k=5, weights=w).fit(Xr, yr)
    pred = reg.predict(Xr_test)
    mae = np.mean(np.abs(pred - yr_test))
    print(f"  {w:>8s}  MAE = {mae:.4f}")
    print("    预测前5:", np.round(pred[:5], 3), " 真实前5:", yr_test[:5])

# ============ 7. 模型评估 ============
labels = sorted(np.unique(y_test_np))
acc = accuracy_score(y_test_np, y_pred)
cm = confusion_matrix(y_test_np, y_pred, labels)
acc2, prec, rec, f1, sup = classification_report(y_test_np, y_pred, labels)

print("\n模型评估：")
print(f"准确率 = {acc:.4f}")
print("混淆矩阵（行=真实，列=预测）：")
print("  " + "".join(f"{lab:>12s}" for lab in labels))
for i, lab in enumerate(labels):
    print(f"  {lab:<10s}" + "".join(f"{v:>12d}" for v in cm[i]))

print(f"\n  {'类别':<12s}{'precision':>10s}{'recall':>10s}{'f1':>10s}{'support':>9s}")
for i, lab in enumerate(labels):
    print(f"  {lab:<12s}{prec[i]:>10.4f}{rec[i]:>10.4f}{f1[i]:>10.4f}{sup[i]:>9d}")
print(f"  {'macro avg':<12s}{prec.mean():>10.4f}{rec.mean():>10.4f}{f1.mean():>10.4f}{sup.sum():>9d}")

assert abs(acc - acc2) < 1e-12
print("\n校验：两条路径算出的准确率一致")
from knn import cross_validate_accuracy

k_values = list(range(1, 32, 2))
cv_mean, cv_std = cross_validate_accuracy(X_train_np, y_train_np, k_values,
                                          n_splits=5, seed=SEED)
best = k_values[int(np.argmax(cv_mean))]

print("\nk 值选择（5 折交叉验证）：")
print(f"  {'k':>3s}  {'平均准确率':>10s}  {'标准差':>8s}  曲线")
for k, m, s in zip(k_values, cv_mean, cv_std):
    mark = "  <-- 最优" if k == best else ""
    print(f"  {k:>3d}  {m:>10.4f}  {s:>8.4f}  {'#' * int(round(m * 40))}{mark}")

print(f"\n较优 k = {best}（交叉验证平均准确率 {cv_mean.max():.4f}）")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import predict_mesh

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============ 8. 可视化：散点图 + 决策边界 + 误分类样本 ============
f1, f2 = 2, 3
X2_train = X_train_np[:, [f1, f2]]
X2_test = X_test_np[:, [f1, f2]]
clf2 = KNNClassifier(k=5).fit(X2_train, y_train_np)
y_pred2 = clf2.predict(X2_test)

pad = 0.5
x_min, x_max = X2_train[:, 0].min() - pad, X2_train[:, 0].max() + pad
y_min, y_max = X2_train[:, 1].min() - pad, X2_train[:, 1].max() + pad
XX, YY = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
Z = predict_mesh(X2_train, y_train_np, XX, YY, k=5)

classes = sorted(set(y_train_np))
colors = ["#4C72B0", "#DD8452", "#55A868"]
Z_idx = np.searchsorted(classes, Z)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.contourf(XX, YY, Z_idx, levels=[-0.5, 0.5, 1.5, 2.5],
            cmap=ListedColormap(["#E8F0FA", "#FBEEE2", "#E6F4EA"]), alpha=0.7)
ax.contour(XX, YY, Z_idx, levels=[0.5, 1.5], colors="gray", linewidths=1)
for i, c in enumerate(classes):
    m = y_train_np == c
    ax.scatter(X2_train[m, 0], X2_train[m, 1], c=colors[i], s=45,
               edgecolor="white", label=f"训练集 {c}")
wrong = y_pred2 != y_test_np
ax.scatter(X2_test[wrong, 0], X2_test[wrong, 1], marker="X", s=150, c="crimson",
           edgecolor="black", linewidth=1.2, label=f"误分类样本（{wrong.sum()} 个）")
ax.set_title("二维散点图与决策边界（k = 5）", fontsize=13)
ax.set_xlabel("花瓣长度（Z-score 标准化）")
ax.set_ylabel("花瓣宽度（Z-score 标准化）")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.errorbar(k_values, cv_mean, yerr=cv_std, marker="o", capsize=4,
            color="#4C72B0", label="5 折交叉验证准确率")
ax.axvline(best, color="crimson", linestyle="--", label=f"最优 k = {best}")
ax.set_title("k 值与交叉验证准确率曲线", fontsize=13)
ax.set_xlabel("邻居个数 k")
ax.set_ylabel("准确率")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("visualization.png", dpi=130)
print("\n图已保存：visualization.png")
print("二维模型误分类样本数:", int(wrong.sum()))
from sklearn.neighbors import KNeighborsClassifier

sk = KNeighborsClassifier(n_neighbors=5).fit(X_train_np, y_train_np)
y_sk = sk.predict(X_test_np)
sk_d = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(X_train_np, y_train_np)
y_sk_d = sk_d.predict(X_test_np)

print("\nsklearn 对照实验（K = 5）：")
print(f"  手写 KNN 准确率        = {accuracy_score(y_test_np, y_pred):.4f}")
print(f"  sklearn 准确率         = {accuracy_score(y_test_np, y_sk):.4f}")
print(f"  预测一致比例           = {np.mean(y_pred == y_sk):.4f}"
      f"（{int(np.sum(y_pred == y_sk))}/{len(y_sk)}）")
print(f"  sklearn 距离加权准确率 = {accuracy_score(y_test_np, y_sk_d):.4f}")

diff = np.flatnonzero(y_pred != y_sk)
print(f"  与 sklearn 预测不同的样本数: {len(diff)}")
for i in diff[:5]:
    print(f"    第{i+1}个 真实={y_test_np[i]} 手写={y_pred[i]} sklearn={y_sk[i]}")
import time
from knn import SafeKNNClassifier, predict_fast

print("\n异常处理演示：")
cases = [
    ("空数据",        lambda: SafeKNNClassifier(k=5).fit([], [])),
    ("维度不一致",    lambda: SafeKNNClassifier(k=5).fit(X_train_np, y_train_np).predict(X_test_np[:, :2])),
    ("k 大于样本数",  lambda: SafeKNNClassifier(k=999).fit(X_train_np, y_train_np)),
    ("X/y 长度不符",  lambda: SafeKNNClassifier(k=5).fit(X_train_np, y_train_np[:-1])),
]
for name, fn in cases:
    try:
        fn()
        print(f"  [{name}] 没有报错（不应该）")
    except ValueError as e:
        print(f"  [{name}] ValueError: {e}")

print("\n性能对比（重复 50 次，每次预测 45 个测试样本）：")
clf_safe = SafeKNNClassifier(k=5).fit(X_train_np, y_train_np)

t0 = time.perf_counter()
for _ in range(50):
    p_loop = clf_safe.predict(X_test_np)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(50):
    p_fast = predict_fast(X_train_np, y_train_np, X_test_np, k=5)
t_fast = time.perf_counter() - t0

print(f"  循环版   耗时 = {t_loop:.3f} 秒")
print(f"  向量化版 耗时 = {t_fast:.3f} 秒")
print(f"  加速比   = {t_loop / t_fast:.1f} 倍")
print(f"  两者预测结果相同: {np.array_equal(p_loop, p_fast)}")
