"""Wave-59 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-59 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 342..347):
- ``emd_hht``           — Huang EMD + Hilbert marginal spectrum
- ``gallant_snp``       — Gallant-Nychka SNP density
- ``srisk``             — Acharya-Engle-Richardson SRISK/MES
- ``wavelet_coherence`` — Torrence-Compo Morlet coherence
- ``tar_coint``         — Balke-Fomby threshold cointegration
- ``extreme_qr``        — Chernozhukov extremal quantile
                          regression
"""

from __future__ import annotations

import math

_SEED = 20261231
_EMD_HHT_SEED = _SEED + 342
_GALLANT_SNP_SEED = _SEED + 343
_SRISK_SEED = _SEED + 344
_WAVELET_COHERENCE_SEED = _SEED + 345
_TAR_COINT_SEED = _SEED + 346
_EXTREME_QR_SEED = _SEED + 347

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


def _isinstance_floats(raw: object) -> dict[str, float]:
    """Float-coerce a lane blob, dropping non-numeric entries and
    any key carrying a forbidden headline metric token."""
    if not isinstance(raw, dict):
        return {}
    return _finite_blob(
        {
            k: float(v)
            for k, v in raw.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and _FORBIDDEN.isdisjoint(k.lower().split("_"))
        }
    )


def bench_emd_hht() -> dict[str, float]:
    try:
        from quant_fund.models.emd_hht import bench_emd as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_EMD_HHT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_gallant_snp() -> dict[str, float]:
    try:
        from quant_fund.models.gallant_snp import (
            bench_gallant_snp as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_GALLANT_SNP_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_srisk() -> dict[str, float]:
    try:
        from quant_fund.models.srisk import bench_srisk as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SRISK_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_wavelet_coherence() -> dict[str, float]:
    try:
        from quant_fund.models.wavelet_coherence import (
            bench_wavelet_coherence as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_WAVELET_COHERENCE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_tar_coint() -> dict[str, float]:
    try:
        from quant_fund.models.tar_coint import bench_tar_coint as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_TAR_COINT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_extreme_qr() -> dict[str, float]:
    try:
        from quant_fund.models.extreme_qr import bench_extreme_qr as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_EXTREME_QR_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
