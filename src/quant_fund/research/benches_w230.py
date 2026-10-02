"""Wave-230 adapters: computational-geometry canon — ear-clipping
triangulation, Sutherland-Hodgman clipping, segment intersection,
point-in-polygon, closest pair, rotating calipers — SYNTHETIC geometric
verification benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.closest_pair import bench_closest_pair
from quant_fund.models.ear_clipping import bench_ear_clipping
from quant_fund.models.point_in_polygon import bench_point_in_polygon
from quant_fund.models.rotating_calipers import bench_rotating_calipers
from quant_fund.models.segment_intersection import bench_segment_intersection
from quant_fund.models.sutherland_hodgman import bench_sutherland_hodgman

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_closest_pair_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("closest_pair", bench_closest_pair(seed=_SEED + 1070)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"closest_pair bench failed: {exc}") from exc


def bench_ear_clipping_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ear_clipping", bench_ear_clipping(seed=_SEED + 1071)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ear_clipping bench failed: {exc}") from exc


def bench_point_in_polygon_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("point_in_polygon", bench_point_in_polygon(seed=_SEED + 1072)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"point_in_polygon bench failed: {exc}") from exc


def bench_rotating_calipers_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("rotating_calipers", bench_rotating_calipers(seed=_SEED + 1073))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rotating_calipers bench failed: {exc}") from exc


def bench_segment_intersection_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("segment_intersection", bench_segment_intersection(seed=_SEED + 1074))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"segment_intersection bench failed: {exc}") from exc


def bench_sutherland_hodgman_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("sutherland_hodgman", bench_sutherland_hodgman(seed=_SEED + 1075))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sutherland_hodgman bench failed: {exc}") from exc
