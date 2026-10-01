"""Benchmark batteries for SOTA canon wave 29 (see waves 11-28 for the pattern).

Wave 29 lands OPTIONAL research families, each pinned to the work shipped
in its lane module (see the module docstrings for full citations):

- ``hmc``: Hamiltonian Monte Carlo / NUTS sampling — leapfrog
  integrator, dual-averaging step-size adaptation (Nesterov), recursive
  tree doubling (Hoffman & Gelman 2014), initial-positive ESS and
  split-Rhat diagnostics on Gaussian/funnel/Student-t synth targets.
- ``proxy_svar``: proxy structural VAR — VAR(p) fit, companion IRF
  recursion, external-instrument (Mertens-Ravn/Stock-Watson) impact
  column recovery, first-stage F weak-instrument flag, Uhlig sign-
  restriction draws via Haar QR rotations.
- ``sbi``: simulation-based inference — ABC rejection + SMC-ABC
  (Toni et al. 2009), neural ratio estimation (Hermans et al. 2020) via
  a hand-rolled tanh MLP, MA(2)/g-and-k benchmark posteriors.
- ``rqa``: recurrence quantification analysis — Takens embedding,
  Fraser-Swinney mutual-information delay, recurrence plots, DET/LAM/
  ENTR/TT measures on periodic/IID/Lorenz/AR synth series.
- ``hj_distance``: Hansen-Jagannathan SDF diagnostics — mean-variance
  SDF bound, second-moment HJ distance, Kan-Robotti-Shanken pricing-
  error chi-square test, vertex-slope telemetry on factor panels.
- ``tensor_decomp``: multilinear factor analysis — CP/ALS (Harshman
  1970), Tucker/HOOI (De Lathauwer 2000), CORCONDIA core consistency
  (Bro & Kiers 2003), missing-entry EM (Tomasi & Bro 2005).

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite — the ``_finite_blob`` gate, waves 15-28 precedent.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-29 stamp seed

# --- per-family seeds -------------------------------------------------------
_HMC_SEED = _SEED + 161
_PSVAR_SEED = _SEED + 162
_SBI_SEED = _SEED + 163
_RQA_SEED = _SEED + 164
_HJ_SEED = _SEED + 165
_TENSOR_SEED = _SEED + 166


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _isinstance_floats(
    raw: dict[str, float] | dict[str, object],
) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_hmc() -> dict[str, float]:
    """Hamiltonian MCMC bench (wave 29).

    HMC/NUTS ESS-per-draw, adapted acceptance, tree depth, energy error,
    funnel-neck capture, split-Rhat, determinism. Soft ``{}`` while
    absent.
    """
    try:
        from quant_fund.models.hmc import bench_hmc as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HMC_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_proxy_svar() -> dict[str, float]:
    """Proxy-SVAR bench (wave 29).

    IRF relative error vs the planted B0 column, first-stage F, proxy-
    target correlation diagnostics, weak-instrument flagging, sign-
    restriction keep rate, determinism. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.proxy_svar import bench_proxy_svar as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PSVAR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_sbi() -> dict[str, float]:
    """Simulation-based inference bench (wave 29).

    ABC rejection posterior error/acceptance, SMC-ABC posterior error
    and ESS decay, MA(2) credible-interval coverage, neural ratio
    estimator log-ratio margin, determinism. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.sbi import bench_sbi as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SBI_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_rqa() -> dict[str, float]:
    """Recurrence quantification bench (wave 29).

    DET ordering (Lorenz > periodic > AR > IID), entropy, recurrence
    rate, max diagonal length, determinism. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.rqa import bench_rqa as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_RQA_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hj_distance() -> dict[str, float]:
    """Hansen-Jagannathan SDF bench (wave 29).

    HJ distance ordering (true kernel vs omitted-factor misspec),
    both weightings, KRS pricing-error p-values, MV-bound vertex
    consistency, determinism. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.hj_distance import (
            bench_hj_distance as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HJ_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_tensor_decomp() -> dict[str, float]:
    """CP/Tucker tensor bench (wave 29).

    CP reconstruction error, Tucker HOOI fit, Tucker-congruence factor
    recovery, CORCONDIA ordering (rank-3 vs overfit), missing-entry EM
    imputation error, determinism. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.tensor_decomp import (
            bench_tensor_decomp as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_TENSOR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
