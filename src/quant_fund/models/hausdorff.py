"""Hausdorff canon: directed and symmetric Hausdorff distance
between point sets, with KD-tree acceleration and a
modified-robust quantile variant. Bench: exact small examples,
asymmetry of the directed version, monotonicity under subset
inclusion. All SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

FloatArray = NDArray[np.float64]


def directed_hausdorff(A: FloatArray, B: FloatArray) -> tuple[float, int]:
    """max_{a∈A} min_{b∈B} ‖a−b‖; returns (dist, argmax index)."""
    tree = cKDTree(np.asarray(B, dtype=np.float64))
    d, idx = tree.query(np.asarray(A, dtype=np.float64))
    i = int(np.argmax(d))
    return float(d[i]), i


def hausdorff(A: FloatArray, B: FloatArray) -> float:
    """Symmetric Hausdorff distance."""
    d1, _ = directed_hausdorff(A, B)
    d2, _ = directed_hausdorff(B, A)
    return max(d1, d2)


def modified_hausdorff(A: FloatArray, B: FloatArray, q: float = 0.9) -> float:
    """Quantile-robust Hausdorff (replaces max with a quantile —
    outlier-tolerant variant)."""
    tree = cKDTree(np.asarray(B, dtype=np.float64))
    d, _ = tree.query(np.asarray(A, dtype=np.float64))
    return float(np.quantile(d, q))


def bench_hausdorff(seed: int = 20261231) -> dict[str, float]:
    out: dict[str, float] = {}
    A = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    B = np.array([[0.0, 0.5], [2.0, 0.5]])
    d_ab, _ = directed_hausdorff(A, B)
    d_ba, _ = directed_hausdorff(B, A)
    out["synthetic_hausdorff_ab"] = d_ab
    out["synthetic_hausdorff_ba"] = d_ba
    out["synthetic_hausdorff_sym"] = hausdorff(A, B)
    # exact: B ⊂ A → directed(B,A)=0
    out["synthetic_hausdorff_subset_zero"] = hausdorff(
        B[:1], A
    )  # {B0} vs A: max(directed) = max_A min_{B0}
    d_sub, _ = directed_hausdorff(B[:1], A)
    out["synthetic_hausdorff_dir_subset"] = d_sub
    # identity
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(50, 2))
    out["synthetic_hausdorff_self"] = hausdorff(X, X)
    # shift
    X2 = X + np.array([1.0, 0.0])
    out["synthetic_hausdorff_shift"] = hausdorff(X, X2)
    # outlier: robust quantile < symmetric
    X3 = np.vstack([X, [[10.0, 10.0]]])
    out["synthetic_hausdorff_outlier"] = hausdorff(X, X3)
    out["synthetic_hausdorff_robust"] = modified_hausdorff(X3, X, 0.95)
    return out
