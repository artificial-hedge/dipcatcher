"""Benchmark batteries for SOTA canon wave 22 (see waves 11-21 for the pattern).

Wave 22 lands OPTIONAL research families, each pinned to the paper shipped
in its lane module (see the module docstrings for full citations; citation
ids verified against arXiv before implementation):

- ``signature_martingale_test``: expected-signature martingale hypothesis
  test (Chevyrev & Oberhauser 2022, arXiv:1810.10971; signature-moment /
  martingale-validity literature — see module docstring for the verified
  citation set and the corrected speculative ids). Omnibus signature-moment
  scores plus permutation/bootstrap-calibrated p-values and an e-process
  arm; size holds on Brownian/GARCH nulls and power grows with planted
  drift, mean-reversion, and AR(1) alternatives. Composes
  ``models/path_signatures`` via lazy import. Pure numpy/scipy.

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite — the ``_finite_blob`` gate, waves 15-21 precedent.

SHRUNK Monte-Carlo budgets (relative to each lane's own test suite):
module-default bench fixtures; lane modules report their own budgets.
"""

from __future__ import annotations

import math

_SEED = 20261226  # wave-22 stamp seed

# --- signature_martingale_test: expected-signature martingale validity -------
_SMT_SEED = _SEED + 95


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


def bench_signature_martingale_test() -> dict[str, float]:
    """Signature-martingale validity bench (wave 22).

    Thin adapter over the lane module's own
    ``bench_signature_martingale_test``: omnibus size on null ensembles,
    power-vs-drift/OU/AR(1) grids, e-process medians and Ville rates.
    Returns ``{}`` while the lane module is absent or on fail-closed
    rejection.
    """
    try:
        from quant_fund.metrics.signature_martingale_test import (
            bench_signature_martingale_test as _smt_core_bench,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_smt_core_bench(seed=_SMT_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
