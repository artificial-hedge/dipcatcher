"""Vectorized vs row-at-a-time expression evaluation (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 601


class Counter:
    n = 0


def _rowwise(col: np.ndarray, thr: float) -> np.ndarray:
    out = []
    for v in col:
        Counter.n += 1
        if v > thr:
            out.append(v * 2 + 1)
    return np.asarray(out)


def _vectorized(col: np.ndarray, thr: float) -> np.ndarray:
    Counter.n += 1  # one call for the whole batch
    return col[col > thr] * 2 + 1


def bench_vectorized_exec(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    calls_saved = 0.0
    for _ in range(30):
        col = rng.uniform(0, 100, 500)
        thr = rng.uniform(20, 80)
        Counter.n = 0
        a = _rowwise(col, thr)
        row_calls = Counter.n
        Counter.n = 0
        b = _vectorized(col, thr)
        vec_calls = Counter.n
        if np.array_equal(a, b):
            ok += 1
        calls_saved += (row_calls - vec_calls) / max(row_calls, 1)
    return {
        "synthetic_vec_equiv": ok / 30,
        "synthetic_vec_call_reduction": calls_saved / 30,
    }
