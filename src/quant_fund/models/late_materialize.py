"""Late materialization: carry row-ids through filter, fetch columns at end (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 736


def late_materialize(cols: dict[str, np.ndarray], filters: list[tuple[str, float]]) -> list[int]:
    """filters: list of (col, min_val). Return surviving row ids."""
    n = len(next(iter(cols.values())))
    alive = np.arange(n)
    for cname, lo in filters:
        alive = alive[cols[cname][alive] >= lo]
    return [int(i) for i in alive]


def bench_late_materialize(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 400
    cols = {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)}
    ids = late_materialize(cols, [("a", 0.0), ("b", 0.0)])
    # oracle: row-by-row
    expect = [i for i in range(n) if cols["a"][i] >= 0 and cols["b"][i] >= 0]
    ok = float(ids == expect)
    # materialize lazily
    vals = cols["c"][np.array(ids)].sum() if ids else 0.0
    expect_v = sum(cols["c"][i] for i in expect)
    return {"synthetic_late_ids": ok, "synthetic_late_sum": float(abs(vals - expect_v) < 1e-9)}
