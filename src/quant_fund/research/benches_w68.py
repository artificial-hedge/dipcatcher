"""Wave-68 adapter benches — optional scorecard families.

Each adapter calls a SYNTHETIC bench on the wave-68 models and
returns a small metric dict consumed by the scorecard runner.
Adapters are honest by contract: they propagate ``score`` and
label every metric ``synthetic_*``. Any import/shape/exception
failure returns ``{}`` so the family is skipped, never faked.

Families (seeds 20261231 + 396..401):
- ``vix_replication``    — Demeterfi variance-swap replication + corridor IV
- ``mala``               — Roberts-Tweedie Metropolis Langevin MCMC
- ``functional_pca``     — Ramsay-Silverman Karhunen-Loeve decomposition
- ``particle_gibbs``     — Andrieu-Doucet-Holenstein conditional-SMC Gibbs
- ``ripley_k``           — Ripley second-order spatial clustering + CSR envelope
- ``synchrosqueezing``   — Daubechies-Lu-Wu synchrosqueezed wavelet ridges
"""

from __future__ import annotations

import math

_SEED = 20261231
_VIX_REPLICATION_SEED = _SEED + 396
_MALA_SEED = _SEED + 397
_FUNCTIONAL_PCA_SEED = _SEED + 398
_PARTICLE_GIBBS_SEED = _SEED + 399
_RIPLEY_K_SEED = _SEED + 400
_SYNCHROSQUEEZING_SEED = _SEED + 401

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


def bench_vix_replication() -> dict[str, float]:
    try:
        from quant_fund.models.vix_replication import (
            bench_vix_replication as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_VIX_REPLICATION_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_mala() -> dict[str, float]:
    try:
        from quant_fund.models.mala import bench_mala as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MALA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_functional_pca() -> dict[str, float]:
    try:
        from quant_fund.models.functional_pca import (
            bench_functional_pca as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_FUNCTIONAL_PCA_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_particle_gibbs() -> dict[str, float]:
    try:
        from quant_fund.models.particle_gibbs import (
            bench_particle_gibbs as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_PARTICLE_GIBBS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_ripley_k() -> dict[str, float]:
    try:
        from quant_fund.models.ripley_k import bench_ripley_k as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_RIPLEY_K_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_synchrosqueezing() -> dict[str, float]:
    try:
        from quant_fund.models.synchrosqueezing import (
            bench_synchrosqueezing as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SYNCHROSQUEEZING_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
