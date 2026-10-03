"""Monte-Carlo standard-error audit for sim-vs-real divergences.

Every microstructure lane reports ``|sim_arm − real| > threshold`` —
but the sim side is one seed of a stochastic simulator. A divergence
inside the arm's Monte-Carlo noise band is not evidence of a model
gap; a divergence at 20·SE is decisive. ``seed_sweep`` re-runs an arm
over K seeds, computes per-metric mean/SE, and converts each reported
gap into a z-score with a significance verdict.

Design: lanes pass *thunks* — ``arm_builder(seed) -> dict`` returning
the lane's arm result, and ``extractors`` mapping metric names to
``dict -> float | None`` accessors — so the library stays domain-
agnostic and any existing bench can adopt it without refactoring.

The SE uses the population std across seeds divided by sqrt(K) (the
mean's sampling error); metrics that are constant across seeds get SE
0 and only diverge if they structurally cannot reach the real value
(flagged ``inexpressible`` when mean == SE == 0 and |mean − real| > 0).

Verdicts per (arm, metric):
- ``significant_divergence``: |z| >= 3 with SE > 0
- ``within_mc_noise``: |z| < 3
- ``inexpressible``: SE == 0 and mean != real — structural gap
- ``matched``: SE == 0 and mean == real (exact structural agreement)
- ``insufficient_data``: extractor returned None on >half the seeds

Pure computation module — no receipts here; lanes seal their own
``*.v1`` payloads embedding the sweep table when they adopt it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

ArmBuilder = Callable[[int], dict[str, Any]]
Extractor = Callable[[dict[str, Any]], float | None]

MIN_SEEDS = 4
Z_THRESHOLD = 3.0


def sweep_arm(
    builder: ArmBuilder,
    extractors: dict[str, Extractor],
    seeds: range | list[int],
) -> dict[str, dict[str, Any]]:
    """Run ``builder(seed)`` over seeds; per-metric mean/se/n_valid."""
    out: dict[str, dict[str, Any]] = {m: {"values": [], "n_valid": 0} for m in extractors}
    for s in seeds:
        res = builder(s)
        for m, ex in extractors.items():
            v = ex(res)
            if v is not None and np.isfinite(v):
                out[m]["values"].append(float(v))
    for m in extractors:
        vals = np.asarray(out[m]["values"])
        n = len(vals)
        out[m]["n_valid"] = n
        out[m]["mean"] = float(np.mean(vals)) if n else None
        out[m]["se"] = float(np.std(vals, ddof=1) / np.sqrt(n)) if n >= 2 else 0.0
        out[m]["std"] = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
        del out[m]["values"]
    return out


def verdicts(
    swept: dict[str, dict[str, Any]],
    real: dict[str, float],
    n_seeds: int,
) -> dict[str, dict[str, Any]]:
    """Per-metric verdict vs a real-tape value."""
    out: dict[str, dict[str, Any]] = {}
    for m, st in swept.items():
        rv = real.get(m)
        mean, se, n = st["mean"], st["se"], st["n_valid"]
        if rv is None or mean is None:
            out[m] = {"verdict": "insufficient_data", "z": None}
            continue
        if n < max(2, n_seeds // 2):
            out[m] = {"verdict": "insufficient_data", "z": None, "mean": mean, "se": se}
            continue
        gap = mean - rv
        if se == 0.0:
            verdict = "matched" if gap == 0.0 else "inexpressible"
            z = None
        else:
            z = gap / se
            verdict = "significant_divergence" if abs(z) >= Z_THRESHOLD else "within_mc_noise"
        out[m] = {
            "verdict": verdict,
            "z": None if z is None else float(z),
            "mean": mean,
            "se": se,
            "real": rv,
            "gap": float(gap),
            "n_valid": n,
        }
    return out


def mcse_audit(
    arms: dict[str, ArmBuilder],
    extractors: dict[str, Extractor],
    real: dict[str, float],
    seeds: range | list[int] = range(8),
) -> dict[str, Any]:
    """Sweep every arm, then verdict every (arm, metric) pair.

    Returns a flat table plus ``summary`` counts keyed by verdict —
    the headline being how many claimed gaps survive MC noise.
    """
    seed_list = list(seeds)
    if len(seed_list) < MIN_SEEDS:
        raise ValueError(f"need >= {MIN_SEEDS} seeds for a stable SE")
    table: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for name, builder in arms.items():
        swept = sweep_arm(builder, extractors, seed_list)
        v = verdicts(swept, real, len(seed_list))
        table[name] = {"sweep": swept, "verdicts": v}
        for vv in v.values():
            counts[vv["verdict"]] = counts.get(vv["verdict"], 0) + 1
    return {
        "n_seeds": len(seed_list),
        "arms": table,
        "summary": {
            "verdict_counts": counts,
            "significant_share": (
                counts.get("significant_divergence", 0) + counts.get("inexpressible", 0)
            )
            / max(1, sum(counts.values())),
        },
    }


def microstructure_mcse_demo(
    data_dir: Path,
    *,
    seeds: range = range(8),
    horizon: int = 30000,
) -> dict[str, Any]:
    """Adoption demo on two live lanes: which divergences survive MC noise?

    Re-seeds the sim arms of ``cancel_lead`` (1.0s signed drift) and
    ``queue_fate`` (join fill rate) and re-verdicts their headline gaps
    against the real-tape values. The real side is deterministic — one
    replay of the tape — so it needs no sweep.

    Receipt kind ``seed_sweep_demo.v1``, data_label MIXED.
    """
    from quant_fund.microstructure.cancel_lead import (
        lobster_cancel_lead,
        sim_cancel_lead,
    )
    from quant_fund.microstructure.queue_fate import (
        lobster_queue_fate,
        sim_queue_fate,
    )
    from quant_fund.microstructure.split_flow import SplitFlow
    from quant_fund.microstructure.zi_lob_simulator import (
        MarkovRegimeFlow,
        RegimeState,
    )

    msg = data_dir / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = data_dir / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg.exists() or not ob.exists():
        raise FileNotFoundError(f"LOBSTER tape required under {data_dir}")

    def regime_flow(s: int) -> Any:
        return MarkovRegimeFlow(
            states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
            stay_probs=(0.995, 0.985),
            seed=7 + s,
        )

    real_lead = lobster_cancel_lead(msg, ob)
    real_fate = lobster_queue_fate(msg, ob)

    lead = mcse_audit(
        arms={
            "iid": lambda s: sim_cancel_lead(None, seed=s, horizon=horizon),
            "regime": lambda s: sim_cancel_lead(regime_flow(s), seed=s, horizon=horizon),
            "split": lambda s: sim_cancel_lead(SplitFlow(seed=11 + s), seed=s, horizon=horizon),
        },
        extractors={
            "drift_1s": lambda d: (
                d.get("per_horizon", {}).get("1.0s", {}).get("mean_signed_drift_ticks")
            ),
        },
        real={
            "drift_1s": real_lead.get("per_horizon", {})
            .get("1.0s", {})
            .get("mean_signed_drift_ticks")
        },
        seeds=seeds,
    )
    fate = mcse_audit(
        arms={
            "iid": lambda s: sim_queue_fate(None, seed=s, horizon=horizon),
            "regime": lambda s: sim_queue_fate(regime_flow(s), seed=s, horizon=horizon),
            "split": lambda s: sim_queue_fate(SplitFlow(seed=11 + s), seed=s, horizon=horizon),
        },
        extractors={
            "join_fill_rate": lambda d: d.get("per_category", {}).get("join", {}).get("fill_rate"),
        },
        real={"join_fill_rate": real_fate.get("per_category", {}).get("join", {}).get("fill_rate")},
        seeds=seeds,
    )
    payload: dict[str, Any] = {
        "kind": "seed_sweep_demo.v1",
        "schema": 1,
        "tape": {"msg": msg.name, "ob": ob.name, "symbol": "AMZN", "date": "2012-06-21"},
        "lanes": {"cancel_lead.drift_1s": lead, "queue_fate.join_fill": fate},
        "claim": "mcse_verdicts_on_wave22b_divergences",
        "interpretation": (
            "The single-seed divergences reported by cancel_lead and "
            "queue_fate are re-verdicted under an 8-seed sweep of each "
            "sim arm. significant_divergence = |z|>=3 against the arm's "
            "own MC standard error; inexpressible = SE 0 and the arm "
            "cannot reach the real value by construction; "
            "within_mc_noise = the headline gap is not statistically "
            "distinguishable under this seed budget."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
