"""Wave-126 adapters: exec-summary self-supervised + test-time-adaptation canon — byol,
barlow_twins, vicreg, tent_tta, shot_tta, ttt_layer —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.barlow_twins import bench_barlow_twins
from quant_fund.models.byol import bench_byol
from quant_fund.models.shot_tta import bench_shot_tta
from quant_fund.models.tent_tta import bench_tent_tta
from quant_fund.models.ttt_layer import bench_ttt_layer
from quant_fund.models.vicreg import bench_vicreg

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


def bench_byol_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("byol", bench_byol(seed=_SEED + 822)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"byol bench failed: {exc}") from exc


def bench_barlow_twins_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("barlow_twins", bench_barlow_twins(seed=_SEED + 823)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"barlow_twins bench failed: {exc}") from exc


def bench_vicreg_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vicreg", bench_vicreg(seed=_SEED + 824)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vicreg bench failed: {exc}") from exc


def bench_tent_tta_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tent_tta", bench_tent_tta(seed=_SEED + 825)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tent_tta bench failed: {exc}") from exc


def bench_shot_tta_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("shot_tta", bench_shot_tta(seed=_SEED + 826)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"shot_tta bench failed: {exc}") from exc


def bench_ttt_layer_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ttt_layer", bench_ttt_layer(seed=_SEED + 827)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ttt_layer bench failed: {exc}") from exc
