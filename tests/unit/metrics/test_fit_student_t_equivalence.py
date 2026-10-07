"""Golden-baseline numerical-equivalence proof for ``fit_student_t`` (T8).

HONESTY CONTRACT: seeded SYNTHETIC inputs only (never market evidence) and
PROPER SCORES only (CRPS) — no Sharpe/Sortino/Calmar/P&L/NAV and no
live-trading or profitability claim.  ``fit_student_t`` underpins proper
scores (CRPS/pinball/QLIKE/PIT), so this suite guards its numerical fidelity.

Workflow (MANDATORY ORDER):
1. ``uv run python tests/unit/metrics/test_fit_student_t_equivalence.py``
   regenerates ``fit_student_t_golden.json`` from the CURRENT ``fit_student_t``
   and must be COMMITTED BEFORE optimizing ``risk_parametric.py``.
2. After optimization, ``test_fit_student_t_matches_golden_baseline`` asserts
   every output field matches the committed baseline within a STATED per-field
   tolerance (all param tolerances are >=100x TIGHTER than the fit's own
   Nelder-Mead convergence criterion, xatol=fatol=1e-4) and FAILS if any cell
   drifts.
3. ``test_equivalence_check_detects_injected_*_drift`` is a PERMANENT mutant
   harness: it injects a known drift and asserts the equivalence check RAISES,
   proving the proof can fail (a proof never observed to fail is
   indistinguishable from one that cannot fail).

Captured output fields per cell: fitted (nu, mu/loc, sigma/scale), maximized
log-likelihood, mean/sum CRPS of the fitted law against the sample, and fitted
quantiles at fixed levels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats as sstats

from quant_fund.metrics.risk_parametric import fit_student_t
from quant_fund.metrics.scoring import crps_student_t

Array = np.ndarray
GOLDEN_PATH = Path(__file__).with_name("fit_student_t_golden.json")
QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)

# Per-field (rtol, atol).  The fit's own convergence criterion is
# Nelder-Mead xatol = fatol = 1e-4; every tolerance below is >=100x tighter.
FIELD_TOL: dict[str, tuple[float, float]] = {
    "fit_nu": (1e-8, 1e-6),
    "fit_mu": (1e-8, 1e-9),
    "fit_sigma": (1e-8, 1e-9),
    "loglik": (1e-11, 1e-9),
    "crps_mean": (1e-9, 1e-12),
    "crps_sum": (1e-9, 1e-9),
    "quantiles": (1e-8, 1e-9),
}


def _spec(k: int, n: int, nu_true: float, loc: float, scale: float, kind: str) -> dict[str, Any]:
    """One deterministic, seeded golden-grid cell specification."""
    return {
        "id": f"c{k:02d}",
        "seed": 20261007 + 7 * k,
        "n": n,
        "nu_true": nu_true,
        "loc": loc,
        "scale": scale,
        "kind": kind,
    }


def _grid() -> list[dict[str, Any]]:
    """Dense seeded grid over the required axes (nu, n, loc/scale, specials).

    Covers nu in [2.05, 2.2, 3, 5, 10, 30, 100], n in [50, 200, 5000, 20000],
    loc/scale across orders of magnitude, plus heavy-tail and near-degenerate
    samples.  Large-n cells use representative nu (heavy/mid/light) to bound
    cost while exercising every required value.
    """
    nus = [2.05, 2.2, 3.0, 5.0, 10.0, 30.0, 100.0]
    locs = [0.0, 50.0, -200.0, 5.0, -3.0, 100.0, -50.0]
    scales = [0.01, 1.0, 100.0, 0.5, 2.0, 30.0, 0.1]
    cells: list[dict[str, Any]] = []
    k = 0
    for i, nu in enumerate(nus):
        for n in (50, 200):
            cells.append(_spec(k, n, nu, locs[i], scales[i], "student"))
            k += 1
    for n in (5000, 20000):
        cells.append(_spec(k, n, 2.05, 5.0, 3.0, "student"))
        k += 1
    for n in (5000, 20000):
        cells.append(_spec(k, n, 100.0, -8.0, 0.2, "student"))
        k += 1
    cells.append(_spec(k, 5000, 2.05, 2.0, 1.0, "heavy"))
    k += 1
    cells.append(_spec(k, 200, 5.0, 0.0, 1.0, "degenerate"))
    k += 1
    cells.append(_spec(k, 5000, 5.0, 20.0, 2.0, "degenerate"))
    return cells


def _with_outliers(z: Array, rng: np.random.Generator) -> Array:
    """Gross a few points to build an explicit heavy-tail sample."""
    count = max(1, z.size // 100)
    idx = rng.choice(z.size, size=count, replace=False)
    out = z.copy()
    out[idx] *= 30.0
    return out


def _make_sample(spec: dict[str, Any]) -> Array:
    """Deterministic SYNTHETIC loss sample for one cell (seeded per cell)."""
    rng = np.random.default_rng(spec["seed"])
    n = spec["n"]
    kind = spec["kind"]
    if kind == "degenerate":
        jitter = spec["scale"] * 1e-3 * rng.standard_normal(n)
        return spec["loc"] + jitter
    z = rng.standard_t(spec["nu_true"], size=n)
    if kind == "heavy":
        z = _with_outliers(z, rng)
    return spec["loc"] + spec["scale"] * z


def _evaluate_fit(losses: Array) -> dict[str, Any]:
    """Run ``fit_student_t`` and capture every golden output field."""
    fit = fit_student_t(losses)
    nu, mu, sigma = fit["nu"], fit["mu"], fit["sigma"]
    v = np.asarray(losses, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    loglik = float(np.sum(sstats.t.logpdf(v, df=nu, loc=mu, scale=sigma)))
    crps = crps_student_t(v, np.full_like(v, mu), np.full_like(v, sigma), nu)
    quants = sstats.t.ppf(QUANTILE_LEVELS, df=nu, loc=mu, scale=sigma)
    return {
        "fit_nu": float(nu),
        "fit_mu": float(mu),
        "fit_sigma": float(sigma),
        "loglik": loglik,
        "crps_mean": float(np.mean(crps)),
        "crps_sum": float(np.sum(crps)),
        "quantiles": [float(q) for q in quants],
    }


def build_golden() -> dict[str, Any]:
    """Generate the full golden baseline from the CURRENT ``fit_student_t``."""
    cells = []
    for spec in _grid():
        cells.append({**spec, **_evaluate_fit(_make_sample(spec))})
    tolerance = {
        field: {"rtol": rtol, "atol": atol} for field, (rtol, atol) in FIELD_TOL.items()
    }
    return {
        "meta": {
            "description": (
                "GOLDEN BASELINE for fit_student_t numerical-equivalence proof (T8). "
                "Seeded SYNTHETIC inputs; proper scores only (CRPS); no market "
                "evidence and no live-trading/P&L claim."
            ),
            "quantile_levels": list(QUANTILE_LEVELS),
            "field_tolerance": tolerance,
            "convergence_criterion": "Nelder-Mead xatol=fatol=1e-4",
            "n_cells": len(cells),
        },
        "cells": cells,
    }


def _is_close(actual: float, golden: float, rtol: float, atol: float) -> bool:
    return abs(actual - golden) <= atol + rtol * abs(golden)


def assert_cell_matches(golden: dict[str, Any], current: dict[str, Any]) -> None:
    """Raise AssertionError if ANY output field drifts beyond its tolerance."""
    for field, (rtol, atol) in FIELD_TOL.items():
        g_val, c_val = golden[field], current[field]
        if field == "quantiles":
            for j, (gv, cv) in enumerate(zip(g_val, c_val)):
                if not _is_close(cv, gv, rtol, atol):
                    raise AssertionError(
                        f"cell {golden['id']} {field}[{j}] drifted: golden={gv!r} current={cv!r}"
                    )
        elif not _is_close(float(c_val), float(g_val), rtol, atol):
            raise AssertionError(
                f"cell {golden['id']} {field} drifted: golden={g_val!r} current={c_val!r} "
                f"(rtol={rtol}, atol={atol})"
            )


def test_fit_student_t_matches_golden_baseline() -> None:
    """Equivalence proof: every cell matches the committed baseline."""
    golden = json.loads(GOLDEN_PATH.read_text())
    for gcell in golden["cells"]:
        spec = {key: gcell[key] for key in ("id", "seed", "n", "nu_true", "loc", "scale", "kind")}
        current = {**spec, **_evaluate_fit(_make_sample(spec))}
        assert_cell_matches(gcell, current)


def test_equivalence_check_detects_injected_param_drift() -> None:
    """PERMANENT mutant harness: a nu drift of 1e-4 (the convergence criterion)
    must be caught — proving the equivalence tolerance is tighter and the proof
    can fail."""
    golden = json.loads(GOLDEN_PATH.read_text())
    gcell = dict(golden["cells"][0])
    drifted = dict(gcell)
    drifted["fit_nu"] = gcell["fit_nu"] + 1e-4
    with pytest.raises(AssertionError, match="fit_nu drifted"):
        assert_cell_matches(gcell, drifted)


def test_equivalence_check_detects_injected_score_drift() -> None:
    """PERMANENT mutant harness: a CRPS drift of 1e-6 relative must be caught."""
    golden = json.loads(GOLDEN_PATH.read_text())
    gcell = dict(golden["cells"][0])
    drifted = dict(gcell)
    drifted["crps_mean"] = gcell["crps_mean"] * (1.0 + 1e-6)
    with pytest.raises(AssertionError, match="crps_mean drifted"):
        assert_cell_matches(gcell, drifted)


if __name__ == "__main__":
    payload = build_golden()
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {payload['meta']['n_cells']} golden cells to {GOLDEN_PATH}")
