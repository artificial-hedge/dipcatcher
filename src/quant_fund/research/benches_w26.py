"""Benchmark batteries for SOTA canon wave 26 (see waves 11-25 for the pattern).

Wave 26 lands OPTIONAL research families, each pinned to the paper shipped
in its lane module (see the module docstrings for full citations; citation
ids verified against arXiv before implementation):

- ``marchenko_pastur``: random-matrix covariance denoising — MP density
  and edge, Tracy-Widom soft threshold, eigenvalue clipping, rotation
  shrinkage (Ledoit-Wolf, Bun-Bouchaud-Potters; see module docstring).
- ``factor_nowcast``: dynamic-factor nowcasting — Kalman filter + RTS
  smoother with ragged-edge NaN masks, Doz-Giannone-Reichlin EM,
  Banbura-Modugno news decomposition (see module docstring).
- ``stationary_bootstrap``: Politis-Romano stationary bootstrap —
  geometric block lengths, percentile CIs, data-driven block length,
  coverage checks on AR(1) (see module docstring).
- ``skill_ratings``: Elo/Glicko-2/Bradley-Terry skill systems — period-
  aggregated Glicko-2 (Illinois volatility solve), sum-preserving Elo,
  Hunter MM Bradley-Terry (see module docstring).
- ``modularity_communities``: Louvain modularity maximization on
  correlation networks — gain-vs-stay local moves, self-loop preserving
  aggregation, ARI/NMI planted-partition recovery (see module docstring).
- ``stein_thinning``: kernel-Stein sample compression — RBF Stein kernel,
  greedy KSD thinning, kernel herding, MMD edges vs random subsets
  (Riabiz, Teymur, Chen-Welling-Smola; see module docstring).

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite — the ``_finite_blob`` gate, waves 15-25 precedent.
"""

from __future__ import annotations

import math

_SEED = 20261230  # wave-26 stamp seed

# --- per-family seeds -------------------------------------------------------
_MP_SEED = _SEED + 131
_DFM_SEED = _SEED + 132
_SB_SEED = _SEED + 133
_SKILL_SEED = _SEED + 134
_MODULARITY_SEED = _SEED + 135
_STEIN_SEED = _SEED + 136


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _isinstance_floats(raw: dict[str, float] | dict[str, object]) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_marchenko_pastur() -> dict[str, float]:
    """Marchenko-Pastur RMT denoising bench (wave 26).

    Signal-count recovery vs factor truth, eigenvalue-clip Frobenius
    improvement, MP-bound violation rate on pure noise. Soft ``{}``
    while absent.
    """
    try:
        from quant_fund.metrics.marchenko_pastur import (
            bench_marchenko_pastur as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MP_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_factor_nowcast() -> dict[str, float]:
    """Dynamic-factor nowcasting bench (wave 26).

    Smoothed-factor correlation vs truth, masked-cell nowcast error vs
    naive carry, EM convergence, news-decomposition sanity. Soft ``{}``
    while absent.
    """
    try:
        from quant_fund.models.factor_nowcast import (
            bench_factor_nowcast as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_DFM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_stationary_bootstrap() -> dict[str, float]:
    """Stationary bootstrap bench (wave 26).

    Percentile-CI coverage on AR(1), width vs iid bootstrap, rho recovery,
    resample index coverage. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.stationary_bootstrap import (
            bench_stationary_bootstrap as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SB_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_skill_ratings() -> dict[str, float]:
    """Skill-ratings bench (wave 26).

    Bradley-Terry log-likelihood improvement, rating-vs-truth Spearman,
    Glicko-2 RD shrink over a rating period, top-1 hit rate. Soft ``{}``
    while absent.
    """
    try:
        from quant_fund.metrics.skill_ratings import (
            bench_skill_ratings as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SKILL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_modularity_communities() -> dict[str, float]:
    """Louvain modularity bench (wave 26).

    ARI/NMI and community-count recovery on planted partitions and
    correlation networks, modularity gain over trivial partitions.
    Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.modularity_communities import (
            bench_modularity_communities as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_MODULARITY_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_stein_thinning() -> dict[str, float]:
    """Stein thinning / herding bench (wave 26).

    MMD and KSD of thinned subsets vs random subsamples, herding edge,
    determinism. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.stein_thinning import (
            bench_stein_thinning as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_STEIN_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
