"""Wave-91 optional scorecard families — the
optimization, clustering, manifold, robust, EB, and
design canon: Nelder-Mead (1965)/Powell (1964)/
Polak-Ribiere nonlinear-CG/BFGS direction-set and
quasi-Newton minimizers plus Levenberg-Marquardt
damped least squares, k-means++ (Arthur-
Vassilvitskii 2007)/PAM (Kaufman-Rousseeuw 1990)/
DBSCAN (Ester et al. 1996)/OPTICS (Ankerst 1999),
LLE (Roweis-Saul 2000)/Laplacian eigenmaps (Belkin-
Niyogi 2003)/diffusion maps (Coifman-Lafon 2006)/
t-SNE (van der Maaten-Hinton 2008), Huber IRLS
(1964)/Tukey S-estimator/LTS (Rousseeuw 1984)/MM
(Yohai 1987) robust regression, Robbins (1956)
Poisson EB/Tweedie f-modeling/Kiefer-Wolfowitz
NPMLE, and 2^k factorial/Plackett-Burman (1946)/
CCD/Box-Behnken/LHS/Fedorov D-optimal design of
experiments. Emitted only when the corresponding
module's `bench_*` self-check completes on its
SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_OPT_SEED = 20261231 + 534
_CLUSTER_SEED = 20261231 + 535
_MANIFOLD_SEED = 20261231 + 536
_ROBUST_SEED = 20261231 + 537
_EB_SEED = 20261231 + 538
_DOE_SEED = 20261231 + 539

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


def bench_unconstrained_optimizers() -> dict[str, float]:
    try:
        from quant_fund.models.unconstrained_optimizers import (
            bench_optimizers as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_OPT_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_clustering_methods() -> dict[str, float]:
    try:
        from quant_fund.models.clustering_methods import (
            bench_cluster as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CLUSTER_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_manifold_learning() -> dict[str, float]:
    try:
        from quant_fund.models.manifold_learning import (
            bench_manifold as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MANIFOLD_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_robust_regression() -> dict[str, float]:
    try:
        from quant_fund.models.robust_regression import (
            bench_robust as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_ROBUST_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_empirical_bayes() -> dict[str, float]:
    try:
        from quant_fund.models.empirical_bayes import (
            bench_eb as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_EB_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_design_experiments() -> dict[str, float]:
    try:
        from quant_fund.models.design_experiments import (
            bench_doe as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_DOE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
