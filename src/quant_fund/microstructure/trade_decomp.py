"""trade_decomp — effective/realized spread decomposition, trade by trade.

Companion to ``spread_decomp`` (Huang–Stoll's λ regression on aggregate
stats): this is the per-trade accounting identity

    effective_half_spread  = 2 * side * (p_exec - m_pre)
    realized_half_spread   = 2 * side * (p_exec - m_{t+Δ})
    price_impact           = effective - realized
                             = 2 * side * (m_{t+Δ} - m_pre)

effective is what the aggressor paid vs the pre-trade mid; realized is
what the liquidity provider *kept* after the market marked out Δ seconds
later; the difference is the adverse-selection cost absorbed by makers —
the quantity that decides whether posting the touch is profitable.

On the LOBSTER tape each EXECUTION row gives the aggressor side
(``-direction``), the exec price, and the mid ladder from the orderbook
file (row i = post-event state, so the pre-event touch is the *previous*
row's mid). On the sim, ``sim.trades`` gives ``(t, aggressor, price)`` and
``sim.mid`` is tracked stepwise.

Ticks: LOBSTER prices are integer 1e-4 dollars so all quantities here are
in ticks already (price int / 1e-4 dollars = tick). Sim prices are in
``level * tick`` — we normalize by ``cfg.tick``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from quant_fund.microstructure.lobster import (
    EXECUTION,
    parse_messages,
    parse_orderbook_row,
)
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = npt.NDArray[np.float64]

HORIZONS_S: tuple[float, ...] = (0.5, 1.0, 5.0, 10.0)


def _decomp_stats(
    execs: list[tuple[float, int, float]],
    mid_times: Array,
    mids: Array,
) -> dict[str, Any]:
    """``execs``: (t, aggressor_sign, exec_price_ticks)."""
    per_horizon: dict[str, Any] = {}
    eff = []
    for t, s, p in execs:
        i0 = int(np.searchsorted(mid_times, t, "left")) - 1
        if i0 < 0:
            continue
        eff.append(2.0 * s * (p - mids[i0]))
    eff_all = np.asarray(eff, dtype=float)
    eff_all = eff_all[np.isfinite(eff_all)]
    for h in HORIZONS_S:
        eff, rea, imp = [], [], []
        for t, s, p in execs:
            i0 = int(np.searchsorted(mid_times, t, "left")) - 1
            if i0 < 0:
                continue
            m0 = mids[i0]
            j = int(np.searchsorted(mid_times, mid_times[i0] + h, "left"))
            if j >= mids.size:
                continue
            e = 2.0 * s * (p - m0)
            rr = 2.0 * s * (p - mids[j])
            eff.append(e)
            rea.append(rr)
            imp.append(e - rr)
        if not eff:
            continue
        eff_a, rea_a, imp_a = map(np.asarray, (eff, rea, imp))
        per_horizon[f"{h}s"] = {
            "n": int(eff_a.size),
            "effective_mean": float(eff_a.mean()),
            "effective_median": float(np.median(eff_a)),
            "realized_mean": float(rea_a.mean()),
            "realized_median": float(np.median(rea_a)),
            "impact_mean": float(imp_a.mean()),
            "impact_share": float(imp_a.mean() / eff_a.mean())
            if abs(float(eff_a.mean())) > 1e-12
            else float("nan"),
            "realized_positive_share": float((rea_a > 0).mean()),
        }
    return {
        "n_execs": len(execs),
        "per_horizon": per_horizon,
        "effective_mean_ticks": float(eff_all.mean()) if eff_all.size else float("nan"),
    }


def lobster_trade_decomp(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    """Per-trade effective/realized/impact decomposition on the tape."""
    events = list(parse_messages(msg_path))
    execs: list[tuple[float, int, float]] = []
    mids: list[float] = []
    with open(ob_path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            asks, bids = parse_orderbook_row(row)
            # prices are integer 1e-4 dollars; /100 converts to ticks
            mids.append((asks[0][0] + bids[0][0]) / 200.0)
    # mid ladder times are the message file's event times (row i = post-event)
    times = [ev.time_s for ev in events]
    for ev in events:
        if ev.event_type == EXECUTION:
            execs.append((ev.time_s, -ev.direction, float(ev.price) / 100.0))
    return _decomp_stats(
        execs,
        np.asarray(times, dtype=float),
        np.asarray(mids[: len(times)], dtype=float),
    )


def sim_trade_decomp(
    flow: Any | None,
    *,
    seed: int = 0,
    horizon: int = 60000,
) -> dict[str, Any]:
    """Same decomposition on the ZI-LOB sim, time in sim steps."""
    cfg = ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    tick = cfg.tick
    execs: list[tuple[float, int, float]] = []
    mid_times: list[float] = []
    mids: list[float] = []
    sim.trades.clear()
    for _ in range(horizon):
        sim.step()
        mid_times.append(sim.t)
        mids.append(sim.mid / tick if sim.mid is not None else np.nan)
        for tr in sim.trades:
            sign = 1 if tr.aggressor == "buy" else -1
            execs.append((tr.t, sign, tr.price / tick))
        sim.trades.clear()
    mids_arr = np.asarray(mids)
    # horizons are in seconds on tape; sim time is steps — map nominal
    # horizon seconds to steps via the mean event rate
    rate = horizon / max(sim.t, 1e-9)
    stats = _decomp_stats_scaled(execs, np.asarray(mid_times), mids_arr, scale=rate)
    stats["note"] = "horizons scaled by the mean event rate (s -> sim steps)"
    return stats


def _decomp_stats_scaled(
    execs: list[tuple[float, int, float]],
    mid_times: Array,
    mids: Array,
    scale: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_execs": len(execs), "per_horizon": {}}
    for h in HORIZONS_S:
        hs = h * scale
        eff, rea, imp = [], [], []
        for t, s, p in execs:
            i0 = int(np.searchsorted(mid_times, t, "left")) - 1
            if i0 < 0 or not np.isfinite(mids[i0]):
                continue
            j = int(np.searchsorted(mid_times, mid_times[i0] + hs, "left"))
            if j >= mids.size or not np.isfinite(mids[j]):
                continue
            e = 2.0 * s * (p - mids[i0])
            rr = 2.0 * s * (p - mids[j])
            eff.append(e)
            rea.append(rr)
            imp.append(e - rr)
        if not eff:
            continue
        eff_a, rea_a, imp_a = map(np.asarray, (eff, rea, imp))
        out["per_horizon"][f"{h}s"] = {
            "n": int(eff_a.size),
            "effective_mean": float(eff_a.mean()),
            "realized_mean": float(rea_a.mean()),
            "impact_mean": float(imp_a.mean()),
            "impact_share": float(imp_a.mean() / eff_a.mean())
            if abs(float(eff_a.mean())) > 1e-12
            else float("nan"),
            "realized_positive_share": float((rea_a > 0).mean()),
        }
    return out


def _divergences(real: dict[str, Any], arms: dict[str, dict[str, Any]]) -> list[str]:
    divs: list[str] = []
    for h in HORIZONS_S:
        r = real.get("per_horizon", {}).get(f"{h}s")
        if not r:
            continue
        for arm, blk in arms.items():
            s = blk.get("per_horizon", {}).get(f"{h}s")
            if not s:
                continue
            for k in ("effective_mean", "realized_mean", "impact_share"):
                rv, sv = r.get(k), s.get(k)
                if rv is None or sv is None:
                    continue
                if not (np.isfinite(rv) and np.isfinite(sv)):
                    continue
                if abs(rv - sv) > 0.15:
                    divs.append(f"{arm}:{h}:{k}")
    return divs


def trade_decomp_bench(data_dir: Path, *, horizon: int = 60000) -> dict[str, Any]:
    """Real tape vs the three flow arms; sealed trade_decomp.v1."""
    msg = data_dir / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = data_dir / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg.exists() or not ob.exists():
        raise FileNotFoundError(f"LOBSTER tape required under {data_dir}")

    real = lobster_trade_decomp(msg, ob)
    arms = {
        "iid": sim_trade_decomp(None, seed=0, horizon=horizon),
        "regime": sim_trade_decomp(
            MarkovRegimeFlow(
                states=(
                    RegimeState("calm", 1.0, 0.5),
                    RegimeState("bursty", 3.0, 0.62),
                ),
                stay_probs=(0.995, 0.985),
                seed=7,
            ),
            seed=0,
            horizon=horizon,
        ),
        "split": sim_trade_decomp(SplitFlow(seed=11), seed=0, horizon=horizon),
    }
    divergences = _divergences(real, arms)
    payload: dict[str, Any] = {
        "kind": "trade_decomp",
        "schema": "trade_decomp.v1",
        "git_revision": git_revision(),
        "data_label": "MIXED",
        "research_only": True,
        "claim": {
            "ticker": "AMZN",
            "date": "2012-06-21",
            "horizons_s": list(HORIZONS_S),
            "real": real,
            "arms": arms,
            "divergences": divergences,
            "n_divergences": len(divergences),
        },
        "interpretation": {
            "effective_vs_realized": "gap = adverse selection absorbed by makers",
            "impact_share": "share of the effective spread that is permanent impact",
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "HORIZONS_S",
    "lobster_trade_decomp",
    "sim_trade_decomp",
    "trade_decomp_bench",
]
