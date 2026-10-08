"""Columnar scan: vectorized predicate + projection over column batches (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 734


def columnar_filter(cols: dict[str, np.ndarray], pred_col: str, lo: float, hi: float) -> np.ndarray:
    """Boolean mask rows where lo <= col <= hi."""
    c = cols[pred_col]
    return (c >= lo) & (c <= hi)


def project_sum(
    cols: dict[str, np.ndarray], mask: np.ndarray, cols_out: list[str]
) -> dict[str, float]:
    return {name: float(cols[name][mask].sum()) for name in cols_out}


def bench_columnar_scan(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 500
    cols = {"a": rng.normal(size=n), "b": rng.normal(size=n)}
    mask = columnar_filter(cols, "a", 0.0, 1.0)
    got = project_sum(cols, mask, ["b"])
    idx = np.where(mask)[0]
    expect = float(cols["b"][idx].sum())
    ok = float(abs(got["b"] - expect) < 1e-9 and mask.sum() == len(idx))
    return {"synthetic_columnar_exact": ok}
