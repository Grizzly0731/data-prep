# -*- coding: utf-8 -*-
import numpy as np


def _pairwise(a, b, reducer):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    A, B = np.atleast_2d(a), np.atleast_2d(b)
    d = reducer(A[:, None, :] - B[None, :, :])
    if a.ndim == 1 and b.ndim == 1:
        return float(d[0, 0])
    if a.ndim == 1:
        return d[0]
    if b.ndim == 1:
        return d[:, 0]
    return d


def euclidean_distance(a, b):
    return _pairwise(a, b, lambda diff: np.sqrt(np.sum(diff ** 2, axis=-1)))


def manhattan_distance(a, b):
    return _pairwise(a, b, lambda diff: np.sum(np.abs(diff), axis=-1))


def find_nearest_neighbors(X_train, x, k=5):
    X_train = np.asarray(X_train, dtype=float)
    dists = euclidean_distance(x, X_train)
    idx = np.argsort(dists, kind="stable")[:k]
    return idx, dists[idx]


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return np.array([self._vote(x) for x in X])

    def _vote(self, x):
        idx, dist = find_nearest_neighbors(self.X_train, x, self.k)
        labels, counts = np.unique(self.y_train[idx], return_counts=True)
        return sorted(labels[counts == counts.max()].tolist())[0]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class KNNRegressor:
    def __init__(self, k=5, weights="uniform"):
        self.k = k
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        idx, dist = find_nearest_neighbors(self.X_train, x, self.k)
        y_nn = self.y_train[idx]
        if self.weights == "uniform":
            return float(np.mean(y_nn))
        w = 1.0 / (dist + 1e-8)
        return float(np.sum(w * y_nn) / np.sum(w))


def accuracy_score(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels):
    labels = list(labels)
    pos = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[pos[t], pos[p]] += 1
    return cm


def classification_report(y_true, y_pred, labels):
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return tp.sum() / cm.sum(), precision, recall, f1, cm.sum(axis=1)
def cross_validate_accuracy(X, y, k_values, n_splits=5, seed=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(X)), n_splits)

    scores = np.zeros((n_splits, len(k_values)))
    for f, val_idx in enumerate(folds):
        mask = np.ones(len(X), dtype=bool)
        mask[val_idx] = False
        clf = KNNClassifier(k=1).fit(X[mask], y[mask])
        for j, k in enumerate(k_values):
            clf.k = k
            scores[f, j] = np.mean(clf.predict(X[val_idx]) == y[val_idx])
    return scores.mean(axis=0), scores.std(axis=0)
def predict_mesh(X_train, y_train, XX, YY, k=5):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    pts = np.c_[XX.ravel(), YY.ravel()]
    dists = euclidean_distance(pts, X_train)
    nn = np.argsort(dists, axis=1, kind="stable")[:, :k]
    classes, codes = np.unique(y_train, return_inverse=True)
    votes = np.zeros((len(pts), len(classes)), dtype=int)
    rows = np.repeat(np.arange(len(pts)), k)
    np.add.at(votes, (rows, codes[nn].ravel()), 1)
    return classes[votes.argmax(axis=1)].reshape(XX.shape)
class SafeKNNClassifier(KNNClassifier):
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] == 0:
            raise ValueError("训练数据为空：X 中没有任何样本")
        if X.shape[0] != len(y):
            raise ValueError(f"样本数不一致：X 有 {X.shape[0]} 行，y 有 {len(y)} 个标签")
        if self.k > X.shape[0]:
            raise ValueError(f"k 过大：k={self.k}，但训练样本只有 {X.shape[0]} 个")
        self.X_train, self.y_train, self.n_features = X, y, X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            raise ValueError("预测数据为空")
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features:
            raise ValueError(f"特征维度不一致：训练时 {self.n_features} 维，预测时 {X.shape[1]} 维")
        return super().predict(X)


def predict_fast(X_train, y_train, X_query, k=5):
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    dists = euclidean_distance(X_query, X_train)
    nn = np.argsort(dists, axis=1, kind="stable")[:, :k]
    classes, codes = np.unique(y_train, return_inverse=True)
    votes = np.zeros((len(X_query), len(classes)), dtype=int)
    rows = np.repeat(np.arange(len(X_query)), k)
    np.add.at(votes, (rows, codes[nn].ravel()), 1)
    return classes[votes.argmax(axis=1)]
