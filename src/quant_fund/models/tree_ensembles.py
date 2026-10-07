"""Tree ensembles from scratch: CART with Gini/MSE splits,
bagged random forest, and gradient boosting on regression
stumps. Synthetic bench gates nonlinear (XOR/checker)
recovery and additive-model fit."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _gini(y: FloatArray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1 - (p**2).sum())


def _mse(y: FloatArray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.var(y))


def _best_split(
    x: FloatArray, y: FloatArray, classification: bool, feats: FloatArray
) -> tuple[int, float, float]:
    """Exhaustive split search over feature midpoints."""
    n, _ = x.shape
    impurity = _gini if classification else _mse
    base = impurity(y)
    best_gain, best_j, best_t = 0.0, -1, 0.0
    for j in feats:
        vals = np.unique(x[:, j])
        if len(vals) < 2:
            continue
        ts = (vals[:-1] + vals[1:]) / 2
        for t in ts:
            left = x[:, j] <= t
            nl = int(left.sum())
            if nl == 0 or nl == n:
                continue
            gain = base - (nl * impurity(y[left]) + (n - nl) * impurity(y[~left])) / n
            if gain > best_gain:
                best_gain, best_j, best_t = gain, int(j), float(t)
    return best_j, best_t, best_gain


class _Node:
    __slots__ = ("j", "t", "left", "right", "value")

    def __init__(self, j: int = -1, t: float = 0.0, value: float = 0.0) -> None:
        self.j = j
        self.t = t
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.value = value


def _leaf_value(y: FloatArray, classification: bool) -> float:
    if classification:
        vals, counts = np.unique(y, return_counts=True)
        return float(vals[np.argmax(counts)])
    return float(y.mean())


def _build(
    x: FloatArray,
    y: FloatArray,
    depth: int,
    max_depth: int,
    min_leaf: int,
    classification: bool,
    mtry: int,
    rng: np.random.Generator,
) -> _Node:
    node = _Node(value=_leaf_value(y, classification))
    n, p = x.shape
    if depth >= max_depth or n < 2 * min_leaf or len(np.unique(y)) == 1:
        return node
    feats = rng.choice(p, size=min(mtry, p), replace=False)
    j, t, gain = _best_split(x, y, classification, feats)
    if j < 0 or gain <= 1e-12:
        return node
    node.j, node.t = j, t
    left = x[:, j] <= t
    node.left = _build(x[left], y[left], depth + 1, max_depth, min_leaf, classification, mtry, rng)
    node.right = _build(
        x[~left], y[~left], depth + 1, max_depth, min_leaf, classification, mtry, rng
    )
    return node


def _predict_one(node: _Node, xi: FloatArray) -> float:
    while node.j >= 0:
        nxt = node.left if xi[node.j] <= node.t else node.right
        if nxt is None:
            break
        node = nxt
    return node.value


def cart_fit(
    x: FloatArray,
    y: FloatArray,
    classification: bool = True,
    max_depth: int = 6,
    min_leaf: int = 2,
    mtry: int | None = None,
    seed: int = 0,
) -> _Node:
    """Single CART tree (Breiman et al. 1984)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if mtry is None:
        mtry = x.shape[1]
    return _build(x, y, 0, max_depth, min_leaf, classification, mtry, rng)


def tree_predict(node: _Node, x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    return np.asarray([_predict_one(node, xi) for xi in x])


def random_forest(
    x: FloatArray,
    y: FloatArray,
    n_trees: int = 50,
    classification: bool = True,
    max_depth: int = 8,
    mtry: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Bagged forest: bootstrap resample + random subspace."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = x.shape
    if mtry is None:
        mtry = max(1, int(np.sqrt(p)))
    trees = []
    oob_idx = []
    for _ in range(n_trees):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        trees.append(_build(x[idx], y[idx], 0, max_depth, 2, classification, mtry, rng))
        oob_idx.append(oob)
    return {"trees": trees, "oob": oob_idx, "classification": classification}


def forest_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    trees = model["trees"]
    if not (isinstance(trees, list)):
        raise ValueError("isinstance(trees, list)")
    preds = np.array([[_predict_one(t, xi) for xi in x] for t in trees])
    classification = bool(model["classification"])
    if classification:
        out = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            _, c = np.unique(preds[:, i], return_counts=True)
            out[i] = np.unique(preds[:, i])[np.argmax(c)]
        return out
    return np.asarray(preds.mean(axis=0))


def gradient_boosting(
    x: FloatArray,
    y: FloatArray,
    n_stumps: int = 100,
    lr: float = 0.1,
    seed: int = 0,
) -> dict[str, object]:
    """Squared-error GBM on depth-1 stumps (Friedman 2001):
    F_m = F_{m-1} + lr·stump(y − F_{m-1})."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    f = np.full(len(y), y.mean())
    stumps = []
    for _ in range(n_stumps):
        r = y - f
        stump = _build(x, r, 0, 1, 2, False, x.shape[1], rng)
        update = np.asarray([_predict_one(stump, xi) for xi in x])
        f += lr * update
        stumps.append(stump)
    return {"init": float(y.mean()), "stumps": stumps, "lr": lr}


def gbm_predict(model: dict[str, object], x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    f = np.full(x.shape[0], float(np.asarray(model["init"])))
    lr = float(np.asarray(model["lr"]))
    stumps = model["stumps"]
    if not (isinstance(stumps, list)):
        raise ValueError("isinstance(stumps, list)")
    for stump in stumps:
        f += lr * np.asarray([_predict_one(stump, xi) for xi in x])
    return f


def bench_trees(seed: int = 546) -> dict[str, float]:
    """SYNTHETIC: two-moons classification — RF must beat a
    single-stump baseline (greedy CART cannot learn pure XOR,
    so the fixture uses a boundary with positive-gain first
    splits); GBM fits an additive signal with small holdout
    MSE."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 400
    t1 = np.linspace(0, np.pi, n // 2)
    moon1 = np.c_[np.cos(t1), np.sin(t1)]
    t2 = np.linspace(0, np.pi, n // 2)
    moon2 = np.c_[np.cos(t2) + 1.0, -np.sin(t2) - 0.4]
    x = np.vstack([moon1, moon2]) + rng.normal(scale=0.08, size=(n, 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(np.int64)
    perm = rng.permutation(n)
    tr, te = perm[:300], perm[300:]
    rf = random_forest(x[tr], y[tr], n_trees=60, classification=True, seed=seed)
    acc = float((forest_predict(rf, x[te]) == y[te]).mean())
    out["synthetic_rf_moons_acc"] = acc
    # single-feature control: x0 alone cannot separate moons
    rf1 = random_forest(x[tr][:, :1], y[tr], n_trees=60, classification=True, seed=seed)
    acc_1 = float((forest_predict(rf1, x[te][:, :1]) == y[te]).mean())
    out["synthetic_rf_moons_1feat_acc"] = acc_1
    if acc < 0.9:
        raise ValueError(f"rf moons acc off: {acc}")
    if acc <= acc_1 + 0.1:
        raise ValueError(f"rf not better than 1-feat: {acc} vs {acc_1}")
    # GBM additive regression
    yr = np.sin(2 * np.pi * x[:, 0]) + x[:, 1] ** 2 - 0.5
    gb = gradient_boosting(x[tr], yr[tr], n_stumps=120, lr=0.2, seed=seed)
    pred = gbm_predict(gb, x[te])
    mse = float(((pred - yr[te]) ** 2).mean())
    out["synthetic_gbm_mse"] = mse
    base_mse = float(yr[te].var())
    out["synthetic_gbm_base_mse"] = base_mse
    if mse > 0.25 * base_mse:
        raise ValueError(f"gbm mse off: {mse} vs base {base_mse}")
    return out
