"""flow_memory — long memory of order flow on the ZI-LOB.

Lillo–Mike–Farmer (2005) and Bouchaud et al.: empirically, the order-sign
sequence is a long-memory process — the variance of the aggregated sign
sum grows superlinearly (Hurst H > 0.5) because institutions split
metaorders. On this sim the mechanism is planted: the 2-state
``MarkovRegimeFlow`` persists direction over runs of market orders, so
the sign sequence should show H > 0.5 under regime flow and H ≈ 0.5
under the iid calm baseline.

Estimator: variance ratio — H = 1 + 0.5·d[log Var(S_k)]/d log k fitted
over a scale grid; plus the first-lag sign autocorrelation as a cheap
complementary stat. Sealed ``flow_memory.v1``, SYNTHETIC.
"""

from __future__ import annotations

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

FLOW_MEMORY_SCHEMA = "flow_memory.v1"


def sign_sequence(
    *,
    config: ZILobConfig,
    horizon: float,
    flow: MarkovRegimeFlow | None = None,
) -> NDArray[np.float64]:
    """Aggressor sign (+1 buy / -1 sell) at each market-order event."""
    sim = ZILobSimulator(config) if flow is None else ZILobSimulator(config, flow=flow)
    out: list[float] = []
    while sim.t < horizon:
        if sim.step() == "market":
            out.append(1.0 if sim.trades[-1].aggressor == "buy" else -1.0)
    return np.asarray(out)


def hurst_var_ratio(
    signs: NDArray[np.float64], scales: tuple[int, ...] = (4, 8, 16, 32, 64)
) -> float:
    """H via aggregated-variance scaling: Var(sum of k signs) ~ k^{2H}.

    H=0.5 for iid signs; H>0.5 for persistent (long-memory) flow.
    """
    n = signs.size
    if n < max(scales) * 8:
        raise ValueError(f"need >= {max(scales) * 8} signs, got {n}")
    ks: list[float] = []
    vs: list[float] = []
    for k in scales:
        m = n // k
        agg = signs[: m * k].reshape(m, k).sum(axis=1)
        if agg.size < 10:
            continue
        v = float(agg.var())
        if v <= 0.0:
            continue
        ks.append(float(k))
        vs.append(v)
    if len(ks) < 3:
        return float("nan")
    slope = float(np.polyfit(np.log(ks), np.log(vs), 1)[0])
    return slope / 2.0


def lag1_autocorr(signs: NDArray[np.float64]) -> float:
    """ρ₁ of the sign sequence."""
    if signs.size < 4:
        return float("nan")
    a, b = signs[:-1], signs[1:]
    return float(np.dot(a, b) / signs.size)


def _flow(seed: int, p_buy: float = 0.78) -> MarkovRegimeFlow:
    return MarkovRegimeFlow(
        states=[
            RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
            RegimeState(name="trend", intensity_mult=1.5, p_buy=p_buy),
        ],
        stay_probs=[0.97, 0.94],
        seed=seed,
    )


def flow_memory_bench(*, n_seeds: int = 8, horizon: float = 15000.0) -> dict[str, Any]:
    """Hurst H of the sign stream: iid baseline vs planted regime flow."""
    arms: dict[str, dict[str, list[float]]] = {
        reg: {"h": [], "rho1": []} for reg in ("calm", "regime")
    }
    for k in range(n_seeds):
        for reg, flow in (("calm", None), ("regime", _flow(9000 + k))):
            signs = sign_sequence(
                config=ZILobConfig(seed=9000 + k, init_depth=8, band=8),
                horizon=horizon,
                flow=flow,
            )
            if signs.size < 256:
                continue
            h = hurst_var_ratio(signs)
            if np.isfinite(h):
                arms[reg]["h"].append(h)
                arms[reg]["rho1"].append(lag1_autocorr(signs))

    def stats(xs: list[float]) -> dict[str, float]:
        a = np.asarray(xs)
        if a.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
        return {"mean": float(a.mean()), "std": float(a.std()), "n": float(a.size)}

    out = {reg: {"hurst": stats(d["h"]), "rho1": stats(d["rho1"])} for reg, d in arms.items()}
    h_calm = out["calm"]["hurst"]["mean"]
    h_reg = out["regime"]["hurst"]["mean"]
    payload: dict[str, Any] = {
        "schema": FLOW_MEMORY_SCHEMA,
        "kind": "flow_memory",
        "n_seeds": n_seeds,
        "horizon": horizon,
        "arms": out,
        "calm_h_near_half": bool(np.isfinite(h_calm) and abs(h_calm - 0.5) < 0.15),
        "regime_h_above_half": bool(np.isfinite(h_reg) and h_reg > 0.5),
        "interpretation": (
            "Aggregated-variance Hurst on the aggressor-sign stream: "
            "planted 2-state flow must register H>0.5 (long memory via "
            "regime persistence); the iid baseline must sit at ~0.5. "
            "Mirrors the Lillo-Mike-Farmer stylized fact that real order "
            "flow is long-memory"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
