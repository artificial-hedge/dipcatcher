"""Wave-236 adapters: graphics canon — raycast DDA, Bresenham,
scanline fill, z-buffer, quaternion slerp, BSP, MVP chain — SYNTHETIC
correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bresenham_line import bench_bresenham_line
from quant_fund.models.bsp_tree import bench_bsp_tree
from quant_fund.models.mvp_transform import bench_mvp_transform
from quant_fund.models.quaternion_slerp import bench_quaternion_slerp
from quant_fund.models.raycaster import bench_raycaster
from quant_fund.models.scanline_fill import bench_scanline_fill
from quant_fund.models.zbuffer_render import bench_zbuffer_render

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


def bench_bresenham_line_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bresenham_line", bench_bresenham_line(seed=_SEED + 1130)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bresenham_line bench failed: {exc}") from exc


def bench_bsp_tree_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bsp_tree", bench_bsp_tree(seed=_SEED + 1131)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bsp_tree bench failed: {exc}") from exc


def bench_mvp_transform_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mvp_transform", bench_mvp_transform(seed=_SEED + 1132)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mvp_transform bench failed: {exc}") from exc


def bench_quaternion_slerp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("quaternion_slerp", bench_quaternion_slerp(seed=_SEED + 1133)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"quaternion_slerp bench failed: {exc}") from exc


def bench_raycaster_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("raycaster", bench_raycaster(seed=_SEED + 1134)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"raycaster bench failed: {exc}") from exc


def bench_scanline_fill_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("scanline_fill", bench_scanline_fill(seed=_SEED + 1135)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"scanline_fill bench failed: {exc}") from exc


def bench_zbuffer_render_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("zbuffer_render", bench_zbuffer_render(seed=_SEED + 1136)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"zbuffer_render bench failed: {exc}") from exc
