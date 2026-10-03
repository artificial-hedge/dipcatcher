"""Bitmap index: per-value bitmaps + bitwise query evaluation."""

import numpy as np

_SEED = 20261231 + 604


class BitmapIndex:
    def __init__(self, col: np.ndarray) -> None:
        self.col = col
        self.maps: dict[int, np.ndarray] = {}
        for v in np.unique(col):
            self.maps[int(v)] = col == v

    def query(self, pred: list[int]) -> np.ndarray:
        m = np.zeros(len(self.col), dtype=bool)
        for v in pred:
            if v in self.maps:
                m |= self.maps[v]
        return np.flatnonzero(m)


def bench_bitmap_index(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(40):
        col = rng.randint(0, 8, 200)
        idx = BitmapIndex(col)
        vs = rng.choice(8, size=rng.randint(1, 4), replace=False).tolist()
        got = idx.query(vs)
        want = np.flatnonzero(np.isin(col, vs))
        if np.array_equal(got, want):
            ok += 1
    return {"synthetic_bitmap_equiv": ok / 40}
