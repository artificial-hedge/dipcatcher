"""Benchmark batteries for SOTA canon wave 23 (see waves 11-22 for the pattern).

Wave 23 lands OPTIONAL research families, each pinned to the paper shipped
in its lane module (see the module docstrings for full citations; citation
ids verified against arXiv before implementation):

- ``svi_surface``: SVI / SSVI implied-volatility-surface parameterization
  and calibration — raw-SVI slice, Gatheral no-butterfly-arbitrage g(k)
  checks, calendar-arbitrage detection + PAVA repair, Roger-Lee wing
  bounds (Gatheral & Jacquier 2014, arXiv:1204.0646; see docstring for
  the corrected companion ids).
- ``propagator_impact``: transient propagator price-impact estimation —
  causal response convolution, power-law kernel fit, half-life /
  permanent-mass diagnostics, expected-cost curves (Bouchaud propagator
  literature — see module docstring).
- ``queue_reactive``: queue-reactive limit-order-book model — CTMC
  birth-death queue simulation, dwell-time MLE, joint stationary law,
  queue-position value curves (Cont & de Larrard queue-reactive
  literature — see module docstring).
- ``koopman_edmd``: Koopman/EDMD nonlinear-dynamics spectra — delay
  embedding, EDMD operator estimation, eigenfunctions, implied
  timescales, regime-separation scores (Koopman/DMD literature — see
  module docstring).
- ``sig_gan``: signature-Wasserstein GAN path generation — SigCWGAN-lite
  residual-GRU generator trained on signature-moment distance with a
  numpy mirror fallback (Ni-Szpruch et al. 2020, arXiv:2006.05421; see
  docstring for corrected Sig-SDE id).
- ``neural_tpp``: transformer neural temporal point process with a
  Hawkes MLE fallback — marked-event simulation, intensity paths,
  compensator integrals (see module docstring).

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite — the ``_finite_blob`` gate, waves 15-22 precedent.

SHRUNK Monte-Carlo budgets (relative to each lane's own test suite):
module-default bench fixtures; lane modules report their own budgets.
"""

from __future__ import annotations

import math

_SEED = 20261227  # wave-23 stamp seed

# --- per-family seeds -------------------------------------------------------
_SVI_SEED = _SEED + 101
_PROP_SEED = _SEED + 102
_QUEUE_SEED = _SEED + 103
_KOOP_SEED = _SEED + 104
_SIGGAN_SEED = _SEED + 105
_NTPP_SEED = _SEED + 106


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _isinstance_floats(raw: dict[str, float] | dict[str, object]) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token
    (``live_pnl_claim`` provenance markers included — the scorecard blob
    carries SYNTHETIC labeling instead)."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_svi_surface() -> dict[str, float]:
    """SVI surface calibration bench (wave 23).

    Thin adapter over the lane module's ``bench_svi_surface``: seeded
    multi-start slice calibration residuals, butterfly/calendar arb
    detection precision-recall, Roger-Lee wing checks, extrapolation
    errors. Returns ``{}`` while the lane module is absent or on
    fail-closed rejection.
    """
    try:
        from quant_fund.models.svi_surface import bench_svi_surface as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SVI_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_propagator_impact() -> dict[str, float]:
    """Transient-propagator impact bench (wave 23).

    Kernel recovery on synthetic impacted series, half-life/permanent-mass
    diagnostics, power-law exponent recovery. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.microstructure.propagator_impact import (
            bench_propagator_impact as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_PROP_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_queue_reactive() -> dict[str, float]:
    """Queue-reactive LOB bench (wave 23).

    Rate-MLE recovery, stationary-law tv distance, joint stationary
    distribution, queue-value curves. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.microstructure.queue_reactive import (
            bench_queue_reactive as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_QUEUE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_koopman_edmd() -> dict[str, float]:
    """Koopman/EDMD spectrum bench (wave 23).

    Eigenvalue/timescale recovery on synthetic oscillators and
    regime-switching systems, forecast horizon scores, regime
    separation AUC. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.koopman_edmd import bench_koopman_edmd as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_KOOP_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_sig_gan() -> dict[str, float]:
    """Signature-GAN generation bench (wave 23).

    Signature-moment distance before/after training on GBM+OU targets,
    marginal KS, ACF distance, torch-availability flag, seeded
    determinism delta. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.sig_gan import bench_sig_gan as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SIGGAN_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_neural_tpp() -> dict[str, float]:
    """Neural temporal point process bench (wave 23).

    Marked-Hawkes parameter recovery via the MLE fallback, intensity-path
    and compensator scores, torch-availability flag. Soft ``{}`` while
    absent.
    """
    try:
        from quant_fund.models.neural_tpp import bench_neural_tpp as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_NTPP_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
