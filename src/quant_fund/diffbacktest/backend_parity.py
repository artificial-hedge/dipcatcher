"""backend_parity — the JAX hard-mode book must equal the NumPy book.

``jax_core`` documents hard mode as "the NumPy-parity book"; this lane makes
that a measured claim. For each strategy and a seeded draw of price matrices
and parameter books, both engines run and every produced array (weights,
turnover, costs, net, nav) must agree elementwise within float64 noise.

If ``jax`` is not installed the bench reports ``status: skipped`` rather than
failing — parity is unverifiable without both backends, and the receipt says
so explicitly instead of fabricating a pass.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from quant_fund.diffbacktest import numpy_core
from quant_fund.diffbacktest.spec import STRATEGIES, StrategyParams
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["backend_parity_bench", "parity_check"]

_TOL = 1e-9
_ARRAY_KEYS = ("weights", "turnover", "costs", "net", "nav")


def parity_check(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams,
    *,
    tol: float = _TOL,
) -> dict[str, Any]:
    """Max |numpy − jax(hard)| per array for one (prices, strategy, params)."""
    from quant_fund.diffbacktest.jax_core import simulate_jax

    np_sim = numpy_core.simulate(prices, strategy, params)
    jx_sim = simulate_jax(prices, strategy, params, mode="hard")
    diffs: dict[str, float] = {}
    for key in _ARRAY_KEYS:
        a = np.asarray(getattr(np_sim, key), dtype=np.float64)
        b = np.asarray(getattr(jx_sim, key), dtype=np.float64)
        diffs[key] = float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else float("inf")
        if a.shape != b.shape:
            diffs[key] = float("inf")
    return {
        "strategy": strategy,
        "max_abs_diff": max(diffs.values()),
        "per_array": diffs,
        "parity": all(d <= tol for d in diffs.values()),
    }


def _params_for(strategy: str, rng: np.random.Generator) -> StrategyParams:
    """A deterministic random draw inside each active parameter's BOX."""
    from quant_fund.diffbacktest.spec import BOXES, active_parameters, harden

    overrides: dict[str, Any] = {}
    for name in active_parameters(strategy):
        lo, hi = BOXES[name]
        overrides[name] = float(rng.uniform(lo, hi))
    base = replace(StrategyParams(), **overrides)
    # lookback must strictly exceed skip on the momentum family.
    if strategy in {"tsmom", "topk", "antonacci", "momentum", "reversal"}:
        lo, hi = BOXES["lookback"]
        base = replace(base, lookback=float(max(hi * 0.6, base.skip + 4)))
    # tsmom: lookback must cover the vol window.
    if strategy == "tsmom":
        base = replace(base, vol_lookback=float(min(base.vol_lookback, base.lookback - 2)))
    return harden(base)


def backend_parity_bench(n_prices: int = 4, seed: int = 0, tol: float = _TOL) -> dict[str, Any]:
    """All strategies × seeded price draws; worst-case per-strategy diff."""
    from quant_fund.diffbacktest.jax_core import available

    if not available():
        return {
            "kind": "backend_parity",
            "schema": "backend_parity.v1",
            "git_revision": git_revision(),
            "data_label": "SYNTHETIC",
            "research_only": True,
            "live_pnl_claim": False,
            "claim": {"invariant": "jax hard mode == numpy book", "status": "skipped"},
            "interpretation": "jax extra not installed — parity unverifiable, not passed",
        }

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for strategy in STRATEGIES:
        strategy_worst = 0.0
        for draw in range(n_prices):
            n_assets = 2 if strategy == "antonacci" else int(rng.integers(3, 8))
            prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (160, n_assets)), axis=0))
            params = _params_for(strategy, rng)
            row = parity_check(prices, strategy, params, tol=tol)
            row["draw"] = draw
            rows.append(row)
            strategy_worst = max(strategy_worst, row["max_abs_diff"])
        worst = max(worst, strategy_worst)
    ok = all(r["parity"] for r in rows)
    payload: dict[str, Any] = {
        "kind": "backend_parity",
        "schema": "backend_parity.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "jax hard mode reproduces the numpy book elementwise",
            "n_strategies": len(STRATEGIES),
            "n_checks": len(rows),
            "tol": tol,
            "worst_abs_diff": worst,
            "status": "ok" if ok else "diverged",
            "divergent": [r["strategy"] for r in rows if not r["parity"]],
        },
        "interpretation": (
            f"{len(rows)} runs across {len(STRATEGIES)} strategies; worst "
            f"|Δ| = {worst:.2e} (tol {tol:g}). "
            + (
                "Backends agree — the differentiable book is the same book."
                if ok
                else "backend divergence — the gradient book trades a different strategy"
            )
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
