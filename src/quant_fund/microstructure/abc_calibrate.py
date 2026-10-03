"""abc_calibrate — ABC rejection calibration of the ZI-LOB simulator.

Fits ``ZILobConfig`` parameters (and the flow *class* — iid vs
metaorder-splitting) against a target summary-statistic vector, by
rejection ABC: draw configs from priors, run the sim at short horizon,
keep the ε-quantile by normalized summary distance.

``sign_lag1`` is deliberately included in the target vector: if the
target has real-tape persistence (~0.72), the ABC posterior will
concentrate on ``use_split=True`` — ABC selecting the model class, not
just parameters. Honest calibration point, not model validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_fund.microstructure.sim_real_ledger import measure_sim
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

ABC_SCHEMA = "abc_calibrate.v1"

# Summary metrics compared sim-to-target; floors prevent div-by-0 when a
# target stat is near zero.
METRICS = ("sign_lag1", "mo_fraction", "spread_ticks_median", "mid_move_std")
FLOORS = {"sign_lag1": 0.05, "mo_fraction": 0.01, "spread_ticks_median": 0.5, "mid_move_std": 0.05}


@dataclass(frozen=True)
class Draw:
    """One ABC proposal: config fields + whether to attach SplitFlow."""

    lam: float
    mu: float
    theta_cxl: float
    band: int
    density_exponent: float
    use_split: bool
    p_start: float
    k_min: int
    intensity_mult: float

    def config(self, seed: int) -> ZILobConfig:
        return ZILobConfig(
            lam=self.lam,
            mu=self.mu,
            theta_cxl=self.theta_cxl,
            band=self.band,
            density_exponent=self.density_exponent,
            seed=seed,
        )

    def flow(self, seed: int) -> SplitFlow | None:
        if not self.use_split:
            return None
        return SplitFlow(
            p_start=self.p_start,
            k_min=self.k_min,
            intensity_mult=self.intensity_mult,
            seed=seed,
        )


def _logu(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def propose(rng: np.random.Generator) -> Draw:
    return Draw(
        lam=_logu(rng, 0.01, 1.0),
        mu=_logu(rng, 0.005, 0.5),
        theta_cxl=_logu(rng, 0.001, 0.2),
        band=int(rng.integers(3, 13)),
        density_exponent=float(rng.uniform(0.0, 2.0)),
        use_split=bool(rng.random() < 0.5),
        p_start=float(rng.uniform(0.02, 0.2)),
        k_min=int(rng.integers(2, 30)),
        intensity_mult=float(rng.uniform(1.0, 4.0)),
    )


def distance(sim_stats: dict[str, Any], target: dict[str, float]) -> float:
    """Normalized Euclidean distance on the shared summary vector."""
    tot = 0.0
    for m in METRICS:
        s = sim_stats.get(m)
        t = target[m]
        if s is None or not math.isfinite(float(s)):
            return float("inf")
        tot += ((float(s) - t) / max(abs(t), FLOORS[m])) ** 2
    return math.sqrt(tot / len(METRICS))


def abc_reject(
    target: dict[str, float],
    *,
    n_draws: int = 64,
    keep: int = 8,
    horizon: int = 3000,
    seed: int = 0,
) -> dict[str, Any]:
    """Rejection ABC: n_draws proposals at short horizon, keep best `keep`."""
    if n_draws < keep or keep < 1:
        raise ValueError(f"need n_draws >= keep >= 1, got {n_draws}/{keep}")
    rng = np.random.default_rng(seed)
    scored: list[tuple[float, Draw, dict[str, Any]]] = []
    for i in range(n_draws):
        d = propose(rng)
        stats = measure_sim(d.config(seed=seed + i), d.flow(seed + 1000 + i), horizon=horizon)
        scored.append((distance(stats, target), d, stats))
    scored.sort(key=lambda t: t[0])
    accepted = scored[:keep]
    return {
        "n_draws": n_draws,
        "keep": keep,
        "horizon": horizon,
        "target": target,
        "accepted": [
            {
                "distance": dist,
                "params": {
                    "lam": d.lam,
                    "mu": d.mu,
                    "theta_cxl": d.theta_cxl,
                    "band": d.band,
                    "density_exponent": d.density_exponent,
                    "use_split": d.use_split,
                    "p_start": d.p_start,
                    "k_min": d.k_min,
                    "intensity_mult": d.intensity_mult,
                },
                "measured": {m: stats.get(m) for m in METRICS},
            }
            for dist, d, stats in accepted
        ],
        "use_split_share": sum(1 for _, d, _ in accepted if d.use_split) / keep,
        "min_distance": accepted[0][0],
        "median_draw_distance": float(np.median([s[0] for s in scored])),
    }


def abc_calibrate_bench(
    target: dict[str, float],
    *,
    n_draws: int = 64,
    keep: int = 8,
    horizon: int = 3000,
    seed: int = 0,
) -> dict[str, Any]:
    res = abc_reject(target, n_draws=n_draws, keep=keep, horizon=horizon, seed=seed)
    best = res["accepted"][0]
    bp = best["params"]
    # Re-measure the best draw at the full ledger horizon for honesty.
    draw = Draw(
        lam=bp["lam"],
        mu=bp["mu"],
        theta_cxl=bp["theta_cxl"],
        band=bp["band"],
        density_exponent=bp["density_exponent"],
        use_split=bp["use_split"],
        p_start=bp["p_start"],
        k_min=bp["k_min"],
        intensity_mult=bp["intensity_mult"],
    )
    full = measure_sim(draw.config(seed + n_draws), draw.flow(seed + 1000 + n_draws), horizon=20000)
    payload: dict[str, Any] = {
        "schema": ABC_SCHEMA,
        "kind": "abc_calibrate",
        "abc": res,
        "refit_full_horizon": {m: full.get(m) for m in METRICS},
        "claims": {
            "split_selected_in_posterior": res["use_split_share"] > 0.5,
            "min_distance_under_median": res["min_distance"] < res["median_draw_distance"],
        },
        "interpretation": (
            "Rejection ABC on the summary vector; the accepted set "
            "concentrates where the sim can actually match the target. "
            "If the target carries real-tape sign persistence, posterior "
            "use_split_share measures how strongly the data select the "
            "splitting mechanism — calibration, not validation"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
