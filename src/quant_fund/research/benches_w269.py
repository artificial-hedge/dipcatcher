"""Wave-269 computer-vision benches: flow, features, geometry, stereo."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.epipolar_8pt import bench_epipolar_8pt
from quant_fund.models.homography_4pt import bench_homography_4pt
from quant_fund.models.lk_flow import bench_lk_flow
from quant_fund.models.orb_feature import bench_orb_feature
from quant_fund.models.ransac_plane import bench_ransac_plane
from quant_fund.models.stereo_disparity import bench_stereo_disparity

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


def bench_lk_flow_family(seed: int = _SEED + 1460) -> dict[str, float]:
    return _floats(_finite_blob("lk_flow", bench_lk_flow(seed)))


def bench_orb_feature_family(seed: int = _SEED + 1461) -> dict[str, float]:
    return _floats(_finite_blob("orb_feature", bench_orb_feature(seed)))


def bench_homography_4pt_family(seed: int = _SEED + 1462) -> dict[str, float]:
    return _floats(_finite_blob("homography_4pt", bench_homography_4pt(seed)))


def bench_ransac_plane_family(seed: int = _SEED + 1463) -> dict[str, float]:
    return _floats(_finite_blob("ransac_plane", bench_ransac_plane(seed)))


def bench_epipolar_8pt_family(seed: int = _SEED + 1464) -> dict[str, float]:
    return _floats(_finite_blob("epipolar_8pt", bench_epipolar_8pt(seed)))


def bench_stereo_disparity_family(seed: int = _SEED + 1465) -> dict[str, float]:
    return _floats(_finite_blob("stereo_disparity", bench_stereo_disparity(seed)))
