# 鸢尾花 KNN 分类

从零实现的 K 近邻（KNN）分类与回归，完成鸢尾花三分类任务，并与 sklearn 对照验证。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `main.py` | 入口：设随机种子、读数据、划分、标准化、分类、回归、评估、可视化 |
| `knn.py` | 算法：欧氏/曼哈顿距离、最近邻搜索、KNNClassifier、KNNRegressor、评估指标 |
| `报告.md` | 完整实验报告（算法流程、结果、参数影响、问题与改进） |

## 运行方式

```bash
python main.py
```

数据文件放在 `data/iris_dataset.csv`。运行后输出各步骤结果，并生成 `visualization.png`。

## 结果

- 测试集准确率 **0.9333**（42/45）
- 交叉验证最优 k = 9，平均准确率 **0.9524**
- 与 sklearn 等权实现 45/45 预测完全一致

## 可复现性

统一使用随机种子 `SEED = 42`：

- 常量定义在 `main.py` 顶部；
- 入口处 `set_seed()` 设置 `random` 与 `numpy` 的种子；
- 数据划分 `df.sample(..., random_state=SEED)`；
- 交叉验证折分 `default_rng(seed)`。

重复运行结果完全一致。

## 依赖

`numpy`、`pandas`、`matplotlib`；对照实验额外使用 `scikit-learn`。
