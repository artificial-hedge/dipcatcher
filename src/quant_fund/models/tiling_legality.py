"""Loop-nest tiling legality via dependence polyhedron (SYNTHETIC).

A rectangular tiling of a nest is legal iff every dependence vector is
componentwise >= 0 (fully permutable nest) — then tile sizes may be
chosen freely and tiles are atomic. We also expose skewing: transform
(i,j) -> (i, i+j) fixes a (1,-1) dependence into (1,0)-form; the
module decides whether a given candidate skew matrix legalizes the nest
and composes tile maps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_SEED = 20261231 + 1032

Dep = tuple[int, ...]


def permutable(deps: list[Dep]) -> bool:
    return all(all(d >= 0 for d in dep) for dep in deps)


def skew(deps: list[Dep], matrix: Any) -> list[Dep]:
    """Apply unimodular skew matrix M to dep vectors: d' = M d."""
    m = np.asarray(matrix, dtype=np.int64)
    out = []
    for dep in deps:
        d = m @ np.asarray(dep, dtype=np.int64)
        out.append(tuple(int(x) for x in d))
    return out


def legalize(deps: list[Dep], candidates: list[Any]) -> Any | None:
    """Return the first candidate matrix making the nest permutable."""
    for m in candidates:
        if permutable(skew(deps, m)):
            return m
    return None


def tile_map(point: tuple[int, ...], sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Tile id containing a point under rectangular sizes."""
    return tuple(p // s for p, s in zip(point, sizes, strict=True))


def dep_crosses_tiles(deps: list[Dep], sizes: tuple[int, ...]) -> bool:
    """True iff some dep may cross a tile boundary (informative metric —
    legal only when combined with permutability)."""
    return any(any(d >= s for d, s in zip(dep, sizes, strict=True)) for dep in deps)


def bench_tiling_legality(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # identity nest permutable -> tile legal at any size
    checks.append(permutable([(1, 0), (0, 1)]))
    # (1,-1) not permutable -> rectangular tiles illegal as-is
    checks.append(not permutable([(1, -1)]))
    # skew (i,j)->(i,i+j): d=(1,-1) -> (1, 0) -> permutable
    m = np.array([[1, 0], [1, 1]])
    checks.append(permutable(skew([(1, -1)], m)))
    # legalize finds a working matrix among candidates
    cands = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [1, 1]])]
    got = legalize([(1, -1)], cands)
    checks.append(got is not None and permutable(skew([(1, -1)], got)))
    # tile map: point (7,5) under (4,4) -> tile (1,1)
    checks.append(tile_map((7, 5), (4, 4)) == (1, 1))
    # dep (1,0) with tiles 4x4 may cross -> informative
    checks.append(dep_crosses_tiles([(4, 0)], (4, 4)))
    # dep (1,0) under 8x8 tiles stays in-tile
    checks.append(not dep_crosses_tiles([(1, 0)], (8, 8)))
    return {"synthetic_tiling_legality": float(sum(checks)) / len(checks)}
