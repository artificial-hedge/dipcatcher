"""Wave-304 computer-vision-2 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.grabcut_lite import bench_grabcut_lite
from quant_fund.models.harris_corner import bench_harris_corner
from quant_fund.models.hough_lines import bench_hough_lines
from quant_fund.models.integral_image import bench_integral_image
from quant_fund.models.meanshift_track import bench_meanshift_track
from quant_fund.models.seam_carving import bench_seam_carving

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


def bench_harris_corner_family(seed: int = _SEED + 1730) -> dict[str, float]:
    return _floats(_finite_blob("harris_corner", bench_harris_corner(seed)))


def bench_hough_lines_family(seed: int = _SEED + 1731) -> dict[str, float]:
    return _floats(_finite_blob("hough_lines", bench_hough_lines(seed)))


def bench_integral_image_family(seed: int = _SEED + 1732) -> dict[str, float]:
    return _floats(_finite_blob("integral_image", bench_integral_image(seed)))


def bench_seam_carving_family(seed: int = _SEED + 1733) -> dict[str, float]:
    return _floats(_finite_blob("seam_carving", bench_seam_carving(seed)))


def bench_grabcut_lite_family(seed: int = _SEED + 1734) -> dict[str, float]:
    return _floats(_finite_blob("grabcut_lite", bench_grabcut_lite(seed)))


def bench_meanshift_track_family(seed: int = _SEED + 1735) -> dict[str, float]:
    return _floats(_finite_blob("meanshift_track", bench_meanshift_track(seed)))
