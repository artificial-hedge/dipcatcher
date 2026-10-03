"""intraday_exec — intraday asymmetry of execution quality.

``intraday_shape`` measured the U in *activity*; ``side_imbalance``
measured fading initiative. This lane asks whether the *cost* of
crossing the spread varies through the session: for every fill,
``effective = 2·s·(p − m_pre)`` and the 5s markout
``realized = 2·s·(p − m_{t+5})`` (the trade_decomp identity) are binned
by session fraction [0,1) between first and last tape event, split by
aggressor side.

The empirical prior: spreads and per-fill impact are elevated near the
open (price discovery) and sometimes into the close — a real tape should
show bucket structure while a ZI-LOB sim with no intraday clock is flat
by construction. Reported per bucket: n, effective/realized means,
buy-initiated share, mean size. The sim runs the identical binning over
its own [0,1) duration; per-bucket divergence flags mark cells the sim
cannot reproduce (not just flat-vs-U — sign/shape).

Sealed ``intraday_exec.v1`` receipt; data_label MIXED when the tape is
present (sim-only arms stay SYNTHETIC inside the same payload).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    EXECUTION,
    parse_messages,
    parse_orderbook_row,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

N_BINS = 10
MARKOUT_S = 5.0


def _bucket_stats(
    execs: list[tuple[float, int, float, int]],
    mid_times: np.ndarray,
    mids: np.ndarray,
    n_bins: int = N_BINS,
) -> list[dict[str, Any]]:
    """execs: (t, aggressor_sign, exec_price_ticks, size)."""
    if not execs:
        return []
    t0, t1 = execs[0][0], execs[-1][0]
    span = max(t1 - t0, 1e-9)
    out: list[dict[str, Any]] = []
    for b in range(n_bins):
        lo, hi = t0 + span * b / n_bins, t0 + span * (b + 1) / n_bins
        eff, rea, buys, sizes = [], [], 0, []
        for t, s, p, sz in execs:
            if not (lo <= t < hi or (b == n_bins - 1 and t == hi)):
                continue
            i0 = int(np.searchsorted(mid_times, t, "left")) - 1
            if i0 < 0:
                continue
            m0 = mids[i0]
            j = int(np.searchsorted(mid_times, mid_times[i0] + MARKOUT_S, "left"))
            if j >= mids.size or not np.isfinite(m0) or not np.isfinite(mids[j]):
                continue
            eff.append(2.0 * s * (p - m0))
            rea.append(2.0 * s * (p - mids[j]))
            buys += int(s > 0)
            sizes.append(sz)
        if not eff:
            continue
        eff_a, rea_a = np.asarray(eff), np.asarray(rea)
        out.append(
            {
                "bin": b,
                "frac_lo": round(b / n_bins, 2),
                "n": int(eff_a.size),
                "effective_mean": float(eff_a.mean()),
                "realized_mean": float(rea_a.mean()),
                "buy_share": float(buys / eff_a.size),
                "mean_size": float(np.mean(sizes)),
            }
        )
    return out


def lobster_intraday_exec(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    events = list(parse_messages(msg_path))
    mids: list[float] = []
    with open(ob_path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            asks, bids = parse_orderbook_row(row)
            mids.append((asks[0][0] + bids[0][0]) / 200.0)
    times = [ev.time_s for ev in events]
    execs = [
        (ev.time_s, -ev.direction, float(ev.price) / 100.0, int(ev.size))
        for ev in events
        if ev.event_type == EXECUTION
    ]
    return {
        "n_execs": len(execs),
        "t_open": times[0] if times else float("nan"),
        "t_close": times[-1] if times else float("nan"),
        "buckets": _bucket_stats(execs, np.asarray(times), np.asarray(mids[: len(times)])),
    }


def sim_intraday_exec(flow: Any | None, *, seed: int = 0, horizon: int = 60000) -> dict[str, Any]:
    cfg = ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    tick = cfg.tick
    execs: list[tuple[float, int, float, int]] = []
    mid_times: list[float] = []
    mids: list[float] = []
    sim.trades.clear()
    for _ in range(horizon):
        sim.step()
        mid_times.append(sim.t)
        mids.append(sim.mid / tick if sim.mid is not None else np.nan)
        for tr in sim.trades:
            execs.append((tr.t, 1 if tr.aggressor == "buy" else -1, tr.price / tick, int(tr.qty)))
        sim.trades.clear()
    rate = horizon / max(sim.t, 1e-9)
    mt = np.asarray(mid_times)
    scaled_mids = np.asarray(mids)
    # map the 5s markout to sim steps via event rate, same trick as trade_decomp
    scaled_times = mt * rate  # sim time -> tape-seconds equivalent
    return {
        "n_execs": len(execs),
        "sim_t": sim.t,
        "buckets": _bucket_stats(
            [(t * rate, s, p, q) for t, s, p, q in execs], scaled_times, scaled_mids
        ),
        "note": "sim time rescaled to tape-seconds via median event rate",
    }


def _divergences(real: list[dict[str, Any]], arms: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for b in real:
        for arm, res in arms.items():
            sim_b = next((x for x in res["buckets"] if x["bin"] == b["bin"]), None)
            if sim_b is None:
                continue
            de = abs(b["effective_mean"] - sim_b["effective_mean"])
            if de > 0.5:
                out.append(
                    f"bin{b['bin']}:{arm}: effective {b['effective_mean']:.2f} vs sim {sim_b['effective_mean']:.2f}"
                )
    return out


def intraday_exec_bench(data_dir: Path, *, horizon: int = 60000) -> dict[str, Any]:
    from quant_fund.microstructure.split_flow import SplitFlow
    from quant_fund.microstructure.zi_lob_simulator import MarkovRegimeFlow, RegimeState

    msg = data_dir / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = data_dir / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not (msg.exists() and ob.exists()):
        raise FileNotFoundError(f"LOBSTER tape missing under {data_dir}")
    real = lobster_intraday_exec(msg, ob)
    arms = {
        "zi_iid": sim_intraday_exec(None, seed=0, horizon=horizon),
        "zi_regime": sim_intraday_exec(
            MarkovRegimeFlow(
                states=(
                    RegimeState("calm", 1.0, 0.5),
                    RegimeState("bursty", 3.0, 0.62),
                ),
                stay_probs=(0.995, 0.985),
                seed=7,
            ),
            seed=1,
            horizon=horizon,
        ),
        "zi_split": sim_intraday_exec(SplitFlow(seed=11), seed=2, horizon=horizon),
    }
    divs = _divergences(real["buckets"], arms)
    tilt = (
        real["buckets"][0]["effective_mean"] - real["buckets"][-1]["effective_mean"]
        if len(real["buckets"]) >= 2
        else float("nan")
    )
    payload: dict[str, Any] = {
        "kind": "intraday_exec",
        "schema": "intraday_exec.v1",
        "git_revision": git_revision(),
        "data_label": "MIXED",
        "research_only": True,
        "claim": {
            "tape": "AMZN 2012-06-21",
            "markout_s": MARKOUT_S,
            "n_bins": N_BINS,
            "real": real,
            "sim_arms": arms,
            "open_minus_close_effective": tilt,
            "divergences": divs,
        },
        "interpretation": {
            "real": "per-bucket effective spread + 5s markout + buy share on the tape",
            "sim": "identical binning on three flow classes (sim time rescaled to seconds)",
            "divergence": "|real - sim| effective spread > 0.5 ticks in a bucket",
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["intraday_exec_bench", "lobster_intraday_exec", "sim_intraday_exec"]
