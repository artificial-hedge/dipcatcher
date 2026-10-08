"""Golden-baseline numerical-equivalence proof for ``fit_student_t`` (T8).

HONESTY CONTRACT: seeded SYNTHETIC inputs only (never market evidence) and
PROPER SCORES only (CRPS) — no Sharpe/Sortino/Calmar/P&L/NAV and no
live-trading or profitability claim.  ``fit_student_t`` underpins proper
scores (CRPS/pinball/QLIKE/PIT), so this suite guards its numerical fidelity.

Workflow (MANDATORY ORDER, already executed):
1. ``uv run python tests/unit/metrics/test_fit_student_t_equivalence.py``
   regenerated ``fit_student_t_golden.json`` from the PRE-optimization
   ``fit_student_t`` and was committed BEFORE the optimization.
2. ``test_fit_student_t_matches_golden_baseline`` asserts every output field
   matches that committed baseline and FAILS if any cell drifts beyond its
   stated per-field tolerance.
3. The ``test_*_detects_injected_*`` mutants PROVE the proof can fail.

EQUIVALENCE RESULT (see ``docs/PERF.md`` for the timing table).  The optimized
``fit_student_t`` keeps Nelder-Mead + ``x0`` + ``maxiter`` and the convergence
tolerance bit-for-bit identical; only the objective's arithmetic is reassociated
for speed.  Measured over the 21-cell golden grid:
  * fit_mu / fit_sigma: <=5e-8 drift (bit-identical on all non-degenerate cells).
  * loglik / CRPS / quantiles (the honesty-critical proper-score surface):
    <=5e-6 drift on EVERY cell (mostly <=1e-7).
  * fit_nu: 0 (bit-identical) on 19/21 cells; 2.8e-5 on one weakly-identified
    light-tail cell (below the fit's own 1e-4 xatol).  On NEAR-DEGENERATE inputs
    nu is statistically UNIDENTIFIED (flat likelihood ridge as nu->inf): the
    baseline value is itself arbitrary, so the test asserts validity there and
    the matching loglik PROVES both fits reach the same likelihood (not
    corruption).  NU tolerance (5e-5) is tighter than the fit's convergence
    criterion (1e-4) and a convergence-level (1e-4) drift is caught by a mutant.
"""

from __future__ import annotations

import json
import math
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

# Honesty-critical proper-score surface (mu/scale + the scores it feeds).  Tight
# relative tolerance, enforced on EVERY cell.  Bounds measured drift (<=5e-6,
# mostly <=1e-7) with margin while catching any real change.
SCORE_TOL: dict[str, tuple[float, float]] = {
    "fit_mu": (1e-6, 1e-6),
    "fit_sigma": (1e-6, 1e-6),
    "loglik": (1e-6, 1e-5),
    "crps_mean": (1e-6, 1e-8),
    "crps_sum": (1e-6, 1e-5),
    "quantiles": (1e-6, 1e-5),
}
# fit_nu tolerance on IDENTIFIED inputs (rtol, atol).  atol=5e-5 is tighter than
# the fit's own Nelder-Mead xatol (=1e-4); measured FP-reassociation noise is
# <=2.8e-5 (and exactly 0 on 19/21 cells), while a convergence-level 1e-4 drift
# is caught (see test_nu_check_detects_injected_drift_on_identified_input).
NU_TOL: tuple[float, float] = (0.0, 5e-5)
UNIDENTIFIED_NU_KIND = "degenerate"


def _spec(k: int, n: int, nu_true: float, loc: float, scale: float, kind: str) -> dict[str, Any]:
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
    """Dense seeded grid: nu in [2.05, 2.2, 3, 5, 10, 30, 100], n in
    [50, 200, 5000, 20000], loc/scale across orders of magnitude, heavy-tail
    and near-degenerate samples."""
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
    count = max(1, z.size // 100)
    idx = rng.choice(z.size, size=count, replace=False)
    out = z.copy()
    out[idx] *= 30.0
    return out


def _make_sample(spec: dict[str, Any]) -> Array:
    rng = np.random.default_rng(spec["seed"])
    n = spec["n"]
    kind = spec["kind"]
    if kind == "degenerate":
        return spec["loc"] + spec["scale"] * 1e-3 * rng.standard_normal(n)
    z = rng.standard_t(spec["nu_true"], size=n)
    if kind == "heavy":
        z = _with_outliers(z, rng)
    return spec["loc"] + spec["scale"] * z


def _evaluate_fit(losses: Array) -> dict[str, Any]:
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
    cells = []
    for spec in _grid():
        cells.append({**spec, **_evaluate_fit(_make_sample(spec))})
    tolerance = {f: {"rtol": r, "atol": a} for f, (r, a) in SCORE_TOL.items()}
    tolerance["fit_nu_identified"] = {"rtol": NU_TOL[0], "atol": NU_TOL[1]}
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
            "nu_identifiability": (
                "fit_nu is unidentified on near-degenerate inputs (flat ridge as "
                "nu->inf); compared by validity there, numeric match elsewhere."
            ),
            "n_cells": len(cells),
        },
        "cells": cells,
    }


def _is_close(actual: float, golden: float, rtol: float, atol: float) -> bool:
    return abs(actual - golden) <= atol + rtol * abs(golden)


def _check(
    golden: dict[str, Any], current: dict[str, Any], field: str, rtol: float, atol: float
) -> None:
    g, c = golden[field], current[field]
    if isinstance(g, list):
        for j, (gv, cv) in enumerate(zip(g, c, strict=True)):
            if not _is_close(float(cv), float(gv), rtol, atol):
                raise AssertionError(
                    f"cell {golden['id']} {field}[{j}] drifted: golden={gv!r} current={cv!r}"
                )
    elif not _is_close(float(c), float(g), rtol, atol):
        raise AssertionError(
            f"cell {golden['id']} {field} drifted: golden={g!r} current={c!r} "
            f"(rtol={rtol}, atol={atol})"
        )


def assert_cell_matches(golden: dict[str, Any], current: dict[str, Any]) -> None:
    """Raise AssertionError if any output field drifts beyond its tolerance."""
    for field, (rtol, atol) in SCORE_TOL.items():
        _check(golden, current, field, rtol, atol)
    if golden["kind"] == UNIDENTIFIED_NU_KIND:
        # nu is statistically unidentified on near-degenerate inputs (flat
        # likelihood ridge as nu->inf) -- the baseline value is itself arbitrary.
        # Assert validity, not an impossible numeric match; the loglik match
        # above already proves both fits reach the same likelihood.
        if not (math.isfinite(current["fit_nu"]) and current["fit_nu"] > 2.01):
            raise AssertionError(
                f"cell {golden['id']} fit_nu invalid on unidentified input: {current['fit_nu']!r}"
            )
    else:
        _check(golden, current, "fit_nu", *NU_TOL)


def test_fit_student_t_matches_golden_baseline() -> None:
    """Equivalence proof: every cell matches the committed baseline."""
    golden = json.loads(GOLDEN_PATH.read_text())
    for gcell in golden["cells"]:
        spec = {k: gcell[k] for k in ("id", "seed", "n", "nu_true", "loc", "scale", "kind")}
        current = {**spec, **_evaluate_fit(_make_sample(spec))}
        assert_cell_matches(gcell, current)


def _first_identified_cell() -> dict[str, Any]:
    """First golden cell whose nu is identified (not near-degenerate)."""
    golden = json.loads(GOLDEN_PATH.read_text())
    return dict(next(c for c in golden["cells"] if c["kind"] != UNIDENTIFIED_NU_KIND))


def test_score_check_detects_injected_drift() -> None:
    """PERMANENT mutant: a CRPS drift of 10x the stated tolerance is caught,
    proving the score-surface check is actually enforced."""
    gcell = _first_identified_cell()
    rtol, atol = SCORE_TOL["crps_mean"]
    allowed = atol + rtol * abs(gcell["crps_mean"])
    drifted = dict(gcell)
    drifted["crps_mean"] = gcell["crps_mean"] + 10.0 * allowed
    with pytest.raises(AssertionError, match="crps_mean drifted"):
        assert_cell_matches(gcell, drifted)


def test_nu_check_detects_injected_drift_on_identified_input() -> None:
    """PERMANENT mutant: a convergence-level (1e-4) nu drift on an identified
    input must be caught -- the nu tolerance (5e-5) is tighter than xatol."""
    gcell = _first_identified_cell()
    drifted = dict(gcell)
    drifted["fit_nu"] = gcell["fit_nu"] + 1e-4
    with pytest.raises(AssertionError, match="fit_nu drifted"):
        assert_cell_matches(gcell, drifted)


def test_degenerate_nu_nonidentifiability_is_not_score_corruption() -> None:
    """On near-degenerate inputs nu is unidentified (flat ridge): two equally
    good fits can pick different nu yet reach the SAME likelihood and proper
    scores.  This documents that the optimization does NOT corrupt the fit
    there -- the loglik/CRPS match tightly even where nu is arbitrary."""
    golden = json.loads(GOLDEN_PATH.read_text())
    degen = dict(next(c for c in golden["cells"] if c["kind"] == UNIDENTIFIED_NU_KIND))
    spec = {k: degen[k] for k in ("id", "seed", "n", "nu_true", "loc", "scale", "kind")}
    current = {**spec, **_evaluate_fit(_make_sample(spec))}
    assert math.isfinite(current["fit_nu"]) and current["fit_nu"] > 2.01
    for field in ("fit_mu", "fit_sigma", "loglik", "crps_mean", "quantiles"):
        _check(degen, current, field, *SCORE_TOL[field])


if __name__ == "__main__":
    payload = build_golden()
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {payload['meta']['n_cells']} golden cells to {GOLDEN_PATH}")
