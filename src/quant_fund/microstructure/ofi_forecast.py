"""Order-flow imbalance (OFI) mid-move forecast on the ZI-LOB simulator.

**Labeled SYNTHETIC** research diagnostic (lane B4): event-level best-quote
OFI per Cont, Kukanov & Stoikov (2014, *Journal of Financial Economics*
12(1):47-88, "The Price Impact of Order Book Events") — the same OFI the
Cont–Cucuringu–Zhang order-flow-imbalance literature regresses mid moves on.
This module measures, per event of ``ZILobSimulator``, the contribution of
best-bid and best-ask updates and asks how well a single event's OFI
predicts the *subsequent* mid change over the next ``horizon`` events.

OFI convention (identical to ``northset.kyle_ofi.cont_ofi_series``, which
operates on static L2 panels — this module is the *event-driven* complement
where the book state comes from the simulator engine, not a snapshot
panel). For each event, with ``D`` = depth at the best quote:

* ``ofi_bid = I(bb up)·D_bb − I(bb down)·D_bb_old + I(bb same)·ΔD_bb``
* ``ofi_ask = I(ba down)·D_ba − I(ba up)·D_ba_old + I(ba same)·ΔD_ba``
* ``ofi = ofi_bid − ofi_ask``

So a new/improved best bid contributes the depth posted there, a depleted
best bid contributes negatively the depth that was there, and an unchanged
best bid contributes its depth change — symmetric on the ask with opposite
sign. A missing best level (one side of the book empty) is treated as a
degraded quote that has disappeared (or a fresh quote appearing), which is
the honest event-level reading; ``n_missing_mid`` counts post-event states
where the mid itself was undefined.

``collect_ofi`` pairs each event's OFI with ``mid_{i+horizon} − mid_i``
(the subsequent k-event mid change; events whose endpoints lack a defined
mid are dropped from the pair stream, not imputed). ``ofi_regression``
fits OLS ``Δmid ~ ofi`` with intercept and reports R² plus OLS and
Bartlett-kernel Newey–West slope standard errors — overlapping horizons
induce MA(horizon) residuals, so the HAC number is the honest one to read.
``ofi_bench`` runs the calm vs trend ``MarkovRegimeFlow`` arms verbatim and
seals an ``ofi_forecast.v1`` receipt (``payload_sha256`` binds the whole
document). Fail-closed: empty/degenerate streams and degenerate
(zero-variance) regression inputs raise ``ValueError``.

Honesty: every output is a SYNTHETIC correctness diagnostic, never market
evidence; claims are booleans computed only from measured values, and both
arms are reported verbatim — no cherry-picking.

References:
- Cont, Kukanov, Stoikov (2014). The price impact of order book events.
  *Journal of Financial Economics* 12(1):47-88 — OFI definition and the
  contemporaneous Δmid ~ OFI regression.
- Cont, Cucuringu, Zhang (2023). Cross-impact of order flow imbalance in
  equity markets. *Quantitative Finance* 23(10):1373-1393.
- Moret, Lillo (2026). Deep learning of robust market making under
  regime-switching order flow. arXiv:2609.11614 — ZI-LOB calibration and
  the MO-clock Markov regime flow used by the trend arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    ZI_LOB_REVISION,
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes, receipt_tree
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]

OFI_FORECAST_SCHEMA = "ofi_forecast.v1"
OFI_FORECAST_KIND = "ofi_forecast"

# Markov stay probabilities shared by both bench arms (on the MO clock).
BENCH_STAY_PROBS = (0.97, 0.94)

__all__ = [
    "BENCH_STAY_PROBS",
    "OFI_FORECAST_KIND",
    "OFI_FORECAST_SCHEMA",
    "OFIStream",
    "collect_ofi",
    "event_ofi",
    "ofi_bench",
    "ofi_regression",
]


def _pos_int(x: int, name: str) -> int:
    if isinstance(x, bool) or int(x) < 1:
        raise ValueError(f"{name} must be an int >= 1, got {x!r}")
    return int(x)


def event_ofi(
    prev_bid_level: int | None,
    prev_bid_depth: int,
    prev_ask_level: int | None,
    prev_ask_depth: int,
    bid_level: int | None,
    bid_depth: int,
    ask_level: int | None,
    ask_depth: int,
) -> tuple[float, float]:
    """Cont–Kukanov–Stoikov per-event OFI contributions ``(ofi_bid, ofi_ask)``.

    Net OFI is ``ofi_bid − ofi_ask``. Depths are the depths *at* the quoted
    best levels (``sim.depth_at``); a ``None`` level means that side of the
    book was empty — a quote that vanished reads as the quote moving away
    (bid down / ask up) and a quote that appeared reads as an improvement
    (bid up / ask down).
    """
    if prev_bid_level is None:
        ofi_bid = 0.0 if bid_level is None else float(bid_depth)
    elif bid_level is None:
        ofi_bid = -float(prev_bid_depth)
    elif bid_level > prev_bid_level:
        ofi_bid = float(bid_depth)
    elif bid_level < prev_bid_level:
        ofi_bid = -float(prev_bid_depth)
    else:
        ofi_bid = float(bid_depth - prev_bid_depth)

    if prev_ask_level is None:
        ofi_ask = 0.0 if ask_level is None else float(ask_depth)
    elif ask_level is None:
        ofi_ask = -float(prev_ask_depth)
    elif ask_level < prev_ask_level:
        ofi_ask = float(ask_depth)
    elif ask_level > prev_ask_level:
        ofi_ask = -float(prev_ask_depth)
    else:
        ofi_ask = float(ask_depth - prev_ask_depth)

    return ofi_bid, ofi_ask


@dataclass(frozen=True)
class OFIStream:
    """Per-event OFI on the ZI-LOB sim, paired with subsequent mid moves.

    ``ofi``/``dmid`` are the aligned regression pair stream: event ``i``'s
    net OFI against ``mid_{i+horizon} − mid_i``, restricted to pairs whose
    endpoints both have a defined mid. ``ofi_events``/``mid`` are the raw
    per-event series (``mid`` is NaN where one side of the book was empty).
    """

    ofi: Array
    dmid: Array
    ofi_events: Array
    mid: Array
    horizon: int
    n_events: int
    n_missing_mid: int

    @property
    def n_pairs(self) -> int:
        return int(self.ofi.size)


def collect_ofi(
    config: ZILobConfig,
    horizon: int,
    *,
    n_events: int = 4000,
    flow: MarkovRegimeFlow | None = None,
) -> OFIStream:
    """Stream (ofi, subsequent k-event mid change) pairs from the ZI-LOB sim.

    Runs ``sim.step()`` ``n_events`` times; event ``i``'s OFI is the book
    change across that step and its label is ``mid_{i+horizon} − mid_i``
    measured on post-event mids. Raises ``ValueError`` on a degenerate
    request (``n_events <= horizon``) or when no pair has a defined mid.
    """
    hz = _pos_int(horizon, "horizon")
    n = _pos_int(n_events, "n_events")
    if n <= hz:
        raise ValueError(f"n_events must exceed horizon to form pairs, got {n} <= {hz}")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")

    sim = ZILobSimulator(config, flow)
    ofi_events = np.empty(n, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        prev_bb, prev_ba = sim.best_bid_level, sim.best_ask_level
        prev_bd = sim.depth_at("buy", prev_bb) if prev_bb is not None else 0
        prev_ad = sim.depth_at("sell", prev_ba) if prev_ba is not None else 0
        sim.step()
        bb, ba = sim.best_bid_level, sim.best_ask_level
        bd = sim.depth_at("buy", bb) if bb is not None else 0
        ad = sim.depth_at("sell", ba) if ba is not None else 0
        ofi_bid, ofi_ask = event_ofi(prev_bb, prev_bd, prev_ba, prev_ad, bb, bd, ba, ad)
        ofi_events[i] = ofi_bid - ofi_ask
        m = sim.mid
        if m is not None:
            mid[i] = m

    dmid_all = mid[hz:] - mid[:-hz]
    ofi_all = ofi_events[:-hz]
    mask = np.isfinite(dmid_all) & np.isfinite(ofi_all)
    ofi = ofi_all[mask]
    dmid = dmid_all[mask]
    if ofi.size == 0:
        raise ValueError("empty OFI stream: no event pair has a defined mid")
    return OFIStream(
        ofi=ofi,
        dmid=dmid,
        ofi_events=ofi_events,
        mid=mid,
        horizon=hz,
        n_events=n,
        n_missing_mid=int(np.isnan(mid).sum()),
    )


def _nw_slope_se(x: Array, resid: Array, lags: int) -> float:
    """Bartlett-kernel Newey–West SE of the OLS slope of ``y ~ 1 + x``."""
    n = int(x.size)
    design = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(design.T @ design)
    xe = design * resid[:, None]
    meat = xe.T @ xe
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1)
        g = xe[lag:].T @ xe[:-lag]
        meat += w * (g + g.T)
    var_beta = xtx_inv @ meat @ xtx_inv
    return float(math.sqrt(max(var_beta[1, 1], 0.0)))


def ofi_regression(
    ofi: Array,
    dmid: Array,
    *,
    nw_lags: int | None = None,
) -> dict[str, Any]:
    """OLS ``dmid ~ ofi`` with intercept: slope, R², OLS and Newey–West SEs.

    Default NW lag follows the Newey–West rule of thumb
    ``floor(4·(n/100)^(2/9))``. Fail-closed: fewer than 3 finite aligned
    observations, non-finite values, or a zero-variance regressor or
    regressand raise ``ValueError`` (a degenerate stream has no measurable
    OFI→mid link to report).
    """
    x = np.asarray(ofi, dtype=np.float64)
    y = np.asarray(dmid, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("ofi and dmid must be aligned 1-D arrays")
    n = int(x.size)
    if n < 3:
        raise ValueError(f"ofi_regression needs >= 3 pairs, got {n}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("ofi and dmid must be finite")
    if float(np.var(x)) <= 0.0:
        raise ValueError("degenerate stream: ofi has zero variance")
    if float(np.var(y)) <= 0.0:
        raise ValueError("degenerate stream: dmid has zero variance")

    design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])
    resid = y - design @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot

    s2 = ss_res / (n - 2)
    xtx_inv = np.linalg.inv(design.T @ design)
    se_ols = float(math.sqrt(s2 * xtx_inv[1, 1]))
    if nw_lags is None:
        lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    else:
        if isinstance(nw_lags, bool) or int(nw_lags) < 0:
            raise ValueError(f"nw_lags must be an int >= 0, got {nw_lags!r}")
        lags = int(nw_lags)
    lags = min(lags, n - 2)
    se_nw = _nw_slope_se(x, resid, lags)
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "slope_se_ols": se_ols,
        "slope_se_nw": se_nw,
        "t_slope_nw": slope / se_nw if se_nw > 0.0 else float("nan"),
        "nw_lags": lags,
        "mean_ofi": float(x.mean()),
        "std_ofi": float(x.std()),
        "mean_dmid": float(y.mean()),
    }


def _bench_arm(
    config: ZILobConfig,
    horizon: int,
    n_events: int,
    states: tuple[RegimeState, RegimeState],
    flow_seed: int,
    nw_lags: int | None,
) -> dict[str, Any]:
    flow = MarkovRegimeFlow(states, BENCH_STAY_PROBS, seed=flow_seed)
    stream = collect_ofi(config, horizon, n_events=n_events, flow=flow)
    reg = ofi_regression(stream.ofi, stream.dmid, nw_lags=nw_lags)
    return {
        "flow": "markov_regime",
        "regime_states": [
            {"name": s.name, "intensity_mult": s.intensity_mult, "p_buy": s.p_buy} for s in states
        ],
        "stay_probs": list(BENCH_STAY_PROBS),
        "flow_seed": int(flow_seed),
        "n_events": int(stream.n_events),
        "horizon": int(stream.horizon),
        "n_pairs": int(stream.n_pairs),
        "n_missing_mid": int(stream.n_missing_mid),
        "n_regime_transitions": len(flow.transitions),
        "regime_mo_counts": [int(c) for c in flow.state_mo_counts],
        "regression": reg,
    }


def ofi_bench(
    config: ZILobConfig | None = None,
    *,
    n_events: int = 4000,
    horizon: int = 25,
    seed: int = 0,
    nw_lags: int | None = None,
) -> dict[str, Any]:
    """Calm vs trend OFI→Δmid bench; returns the sealed ``ofi_forecast.v1`` receipt.

    Both arms use ``MarkovRegimeFlow`` with ``BENCH_STAY_PROBS`` (0.97,
    0.94) on the MO clock. ``calm`` modulates only MO intensity at balanced
    direction (p_buy=0.5); ``trend`` adds a persistent buy-pressure state
    (p_buy=0.75). All arms are reported verbatim; ``claims`` are computed
    only from the measured per-arm R² values.
    """
    n = _pos_int(n_events, "n_events")
    hz = _pos_int(horizon, "horizon")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    cfg = config if config is not None else santa_fe_config(seed=seed)

    arms = {
        "calm": _bench_arm(
            cfg,
            hz,
            n,
            (
                RegimeState("quiet", 0.8, 0.5),
                RegimeState("busy", 1.3, 0.5),
            ),
            seed,
            nw_lags,
        ),
        "trend": _bench_arm(
            cfg,
            hz,
            n,
            (
                RegimeState("base", 1.0, 0.5),
                RegimeState("trend", 1.6, 0.75),
            ),
            seed + 1,
            nw_lags,
        ),
    }

    r2_calm = float(arms["calm"]["regression"]["r2"])
    r2_trend = float(arms["trend"]["regression"]["r2"])
    receipt: dict[str, Any] = {
        "schema": OFI_FORECAST_SCHEMA,
        "kind": OFI_FORECAST_KIND,
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "label": "SYNTHETIC",
        "research_only": True,
        "sim": "zi_lob",
        "sim_revision": ZI_LOB_REVISION,
        "method": "cont_kukanov_stoikov_2014_ofi_ols",
        "config": {
            "s0": cfg.s0,
            "tick": cfg.tick,
            "lam": cfg.lam,
            "mu": cfg.mu,
            "theta_cxl": cfg.theta_cxl,
            "p_buy": cfg.p_buy,
            "band": cfg.band,
            "seed": cfg.seed,
        },
        "n_events": n,
        "horizon": hz,
        "arms": arms,
        "claims": {
            "ofi_r2_positive": bool(r2_calm > 0.0 and r2_trend > 0.0),
            "trend_r2_vs_calm": {
                "trend_r2": r2_trend,
                "calm_r2": r2_calm,
                "delta_r2": r2_trend - r2_calm,
                "trend_higher": bool(r2_trend > r2_calm),
            },
        },
        "note": (
            "Event-level Cont-Kukanov-Stoikov OFI vs subsequent k-event mid "
            "change on the ZI-LOB simulator. SYNTHETIC correctness diagnostic, "
            "never market evidence."
        ),
    }
    _forbidden = ("sharpe", "pnl", "live_pnl_claim")
    leaked = [k for k in receipt if any(tok in k.lower() for tok in _forbidden)]
    if leaked:
        raise AssertionError(f"ofi_forecast bench leaked forbidden research keys: {leaked}")
    payload = {k: v for k, v in receipt.items() if k != "payload_sha256"}
    receipt["payload_sha256"] = hash_bytes(canonical_json_bytes(receipt_tree(payload)))
    return receipt
