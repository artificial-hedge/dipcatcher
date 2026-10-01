"""microprice — Stoikov's depth-weighted mid and its predictive value.

Stoikov (2018) "The micro-price: a high-frequency estimator of future
prices": the imbalance-weighted quote I·a+(1-I)·b (with I the bid share
of touch depth) predicts the next mid move better than the plain mid —
heavy bid queue → price drifts up when it resolves.

This lane streams the ZI-LOB, records touch depth imbalance at each
market-order event together with the realized next-trade mid revision,
and compares the information content of the imbalance signal vs a
sign-only baseline. Claims are measured on the tape, sealed
``microprice.v1``, SYNTHETIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

MICROPRICE_SCHEMA = "microprice.v1"


def touch_imbalance(sim: ZILobSimulator) -> float | None:
    """I = bid_depth / (bid_depth + ask_depth) at the touch; None if empty."""
    bbl, bal = sim.best_bid_level, sim.best_ask_level
    if bbl is None or bal is None:
        return None
    bd = sim.depth_at("buy", bbl)
    ad = sim.depth_at("sell", bal)
    if bd + ad <= 0:
        return None
    return bd / (bd + ad)


@dataclass(frozen=True)
class ImbalanceTape:
    imb: NDArray[np.float64]  # imbalance at each MO event
    dmid: NDArray[np.float64]  # mid revision over the next MO events (ticks)
    signs: NDArray[np.float64]  # aggressor sign (+1 buy)


def collect_imbalance(
    *,
    config: ZILobConfig,
    horizon: float,
    flow: MarkovRegimeFlow | None = None,
    revision_events: int = 5,
) -> ImbalanceTape:
    """At each MO, record touch imbalance and the sign; revision is the mid
    change from this trade to the trade ``revision_events`` later (ticks)."""
    sim = ZILobSimulator(config) if flow is None else ZILobSimulator(config, flow=flow)
    imbs: list[float] = []
    signs: list[float] = []
    mids: list[float] = []
    while sim.t < horizon:
        kind = sim.step()
        if kind != "market":
            continue
        i = touch_imbalance(sim)
        m = sim.mid
        if i is None or m is None:
            continue
        imbs.append(i)
        mids.append(m)
        signs.append(1.0 if sim.trades[-1].aggressor == "buy" else -1.0)
    imb_a = np.asarray(imbs)
    mid_a = np.asarray(mids)
    sign_a = np.asarray(signs)
    h = revision_events
    if mid_a.size <= h:
        return ImbalanceTape(imb_a, np.empty(0), sign_a)
    dmid = (mid_a[h:] - mid_a[:-h]) / config.tick
    return ImbalanceTape(imb_a[:-h], dmid, sign_a[:-h])


def ols_fit(x: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, float]:
    """No-intercept OLS + R² (vs zero predictor) and the sign baseline R²."""
    if x.size < 20:
        return {"beta": float("nan"), "r2": float("nan"), "n": float(x.size)}
    beta = float(np.dot(x, y) / np.dot(x, x))
    resid = y - beta * x
    r2 = 1.0 - float(np.dot(resid, resid) / np.dot(y, y)) if np.dot(y, y) > 0 else float("nan")
    return {"beta": beta, "r2": r2, "n": float(x.size)}


def _flow(seed: int, p_buy: float = 0.78) -> MarkovRegimeFlow:
    return MarkovRegimeFlow(
        states=[
            RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
            RegimeState(name="trend", intensity_mult=1.5, p_buy=p_buy),
        ],
        stay_probs=[0.97, 0.94],
        seed=seed,
    )


def microprice_bench(
    *,
    n_seeds: int = 8,
    horizon: float = 800.0,
    revision_events: int = 5,
) -> dict[str, Any]:
    """Predictive value of touch imbalance vs trade sign, calm vs trend."""
    arms: dict[str, dict[str, list[float]]] = {
        reg: {"r2_imb": [], "r2_sign": [], "beta_imb": []} for reg in ("calm", "trend")
    }
    for k in range(n_seeds):
        for reg, flow in (("calm", None), ("trend", _flow(8000 + k))):
            tape = collect_imbalance(
                config=ZILobConfig(seed=8000 + k, init_depth=8, band=8),
                horizon=horizon,
                flow=flow,
                revision_events=revision_events,
            )
            if tape.dmid.size < 20:
                continue
            fi = ols_fit(tape.imb - 0.5, tape.dmid)
            fs = ols_fit(tape.signs, tape.dmid)
            arms[reg]["r2_imb"].append(fi["r2"])
            arms[reg]["r2_sign"].append(fs["r2"])
            arms[reg]["beta_imb"].append(fi["beta"])

    def stats(xs: list[float]) -> dict[str, float]:
        a = np.asarray([v for v in xs if np.isfinite(v)])
        if a.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
        return {"mean": float(a.mean()), "std": float(a.std()), "n": float(a.size)}

    out = {
        reg: {
            "r2_imbalance": stats(d["r2_imb"]),
            "r2_sign": stats(d["r2_sign"]),
            "beta_imbalance": stats(d["beta_imb"]),
        }
        for reg, d in arms.items()
    }
    imb = out["trend"]["r2_imbalance"]["mean"]
    sig = out["trend"]["r2_sign"]["mean"]
    payload: dict[str, Any] = {
        "schema": MICROPRICE_SCHEMA,
        "kind": "microprice",
        "n_seeds": n_seeds,
        "horizon": horizon,
        "revision_events": revision_events,
        "arms": out,
        "imbalance_beats_sign": bool(np.isfinite(imb) and np.isfinite(sig) and imb > sig),
        "interpretation": (
            "R2 of (imb-0.5) vs sign for the h-event mid revision; the "
            "imbalance signal should dominate under asymmetric queue load "
            "if the sim carries the Stoikov mechanism"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
