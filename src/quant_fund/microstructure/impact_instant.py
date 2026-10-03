"""impact_instant — instantaneous price impact vs trade size.

Cont–Kukanov–Stoikov style: the immediate mid move caused by a fill of
size q. On the real tape sizes span 1 → thousands of shares; the
impact curve E[sign·Δmid | q] is the primal object of market impact.
On the ZI-LOB sim every fill is qty=1 — the size dimension does not
exist — so the honest comparison is the sim's per-unit-lot Δmid
against the real small-size bin, plus the real curve itself.

Measurement: per EXECUTION event i, signed Δmid (ticks) = aggressor
sign × (mid_i − mid_{i−1}) from the orderbook snapshots (ground
truth), joined with the exec size in shares.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import EXECUTION, parse_messages, parse_orderbook_row
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

N_BINS = 8


def _impact_curve(sizes: np.ndarray, signed_dmid: np.ndarray) -> dict[str, Any]:
    """Size-binned impact stats + log-log slope over size for movers."""
    n = sizes.size
    if n < 50:
        return {"ok": False, "n": int(n)}
    qs = np.quantile(sizes, np.linspace(0, 1, N_BINS + 1))
    qs[0], qs[-1] = sizes.min() - 1e-9, sizes.max() + 1e-9
    bins: list[dict[str, Any]] = []
    for lo, hi in zip(qs[:-1], qs[1:], strict=True):
        m = (sizes >= lo) & (sizes < hi)
        if m.sum() == 0:
            continue
        d = signed_dmid[m]
        bins.append(
            {
                "size_lo": float(lo),
                "size_hi": float(hi),
                "n": int(m.sum()),
                "mean_size": float(sizes[m].mean()),
                "mean_signed_dmid_ticks": float(d.mean()),
                "mean_abs_dmid_ticks": float(np.abs(d).mean()),
                "p_move": float(np.mean(d != 0)),
            }
        )
    # log-log slope of mean |Δmid| vs mean size across bins (weighted)
    xs = np.array([b["mean_size"] for b in bins])
    ys = np.array([b["mean_abs_dmid_ticks"] for b in bins])
    ws = np.array([b["n"] for b in bins], dtype=float)
    good = (xs > 0) & (ys > 0)
    alpha = None
    if good.sum() >= 3:
        lx, ly = np.log(xs[good]), np.log(ys[good])
        w = ws[good] / ws[good].sum()
        xm = float((w * lx).sum())
        ym = float((w * ly).sum())
        var = float((w * (lx - xm) ** 2).sum())
        if var > 0:
            alpha = float((w * (lx - xm) * (ly - ym)).sum() / var)
    return {
        "ok": True,
        "n": int(n),
        "mean_signed_dmid_ticks": float(signed_dmid.mean()),
        "mean_abs_dmid_ticks": float(np.abs(signed_dmid).mean()),
        "p_move": float(np.mean(signed_dmid != 0)),
        "size_bins": bins,
        "alpha_loglog": alpha,
        "size_median": float(np.median(sizes)),
        "size_max": float(sizes.max()),
    }


def lobster_impact_instant(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    sizes: list[int] = []
    dmid: list[float] = []
    prev_mid: float | None = None
    with ob_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            asks, bids = parse_orderbook_row(ob_row)
            mid = (asks[0][0] + bids[0][0]) / 200.0 if asks and bids else prev_mid
            if ev.event_type == EXECUTION and mid is not None and prev_mid is not None:
                sizes.append(ev.size)
                dmid.append((-ev.direction) * (mid - prev_mid))
            prev_mid = mid
    return _impact_curve(np.asarray(sizes, dtype=float), np.asarray(dmid))


def sim_impact_instant(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    sizes: list[int] = []
    dmid: list[float] = []
    prev_mid: float | None = None
    n_before = 0
    for _ in range(horizon):
        sim.step()
        mid = sim.mid
        for tr in sim.trades[n_before:]:
            if mid is not None and prev_mid is not None:
                sizes.append(tr.qty)
                dmid.append((1.0 if tr.aggressor == "buy" else -1.0) * (mid - prev_mid))
        n_before = len(sim.trades)
        prev_mid = mid
    return _impact_curve(np.asarray(sizes, dtype=float), np.asarray(dmid))


def impact_instant_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg = sorted(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = sorted(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    if not msg or not ob:
        raise FileNotFoundError(f"no LOBSTER message/orderbook CSV pair under {tape_dir}")
    real = lobster_impact_instant(msg[0], ob[0])
    arms = {
        "iid": sim_impact_instant(seed=seed),
        "regime": sim_impact_instant(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_impact_instant(
            flow=SplitFlow(
                p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 2
            ),
            seed=seed + 2,
        ),
    }
    divergences: list[str] = []
    if real.get("ok"):
        small = next(
            (b for b in real["size_bins"] if b["size_lo"] <= 1.0 < b["size_hi"]),
            real["size_bins"][0] if real["size_bins"] else None,
        )
        if small is not None:
            for name, arm in arms.items():
                if not arm.get("ok"):
                    continue
                r, s = small["mean_abs_dmid_ticks"], arm["mean_abs_dmid_ticks"]
                if abs(r - s) > 0.5:
                    divergences.append(f"{name}_unit_impact_{s:.2f}_vs_{r:.2f}")
        if real["alpha_loglog"] is not None:
            for name, arm in arms.items():
                if arm.get("ok") and arm["alpha_loglog"] is None:
                    divergences.append(f"{name}_no_size_dimension")
    payload: dict[str, Any] = {
        "kind": "impact_instant",
        "schema": "impact_instant.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "instantaneous_impact_curve_measured",
        "interpretation": (
            "Per exec: signed Δmid (aggressor sign × mid change, ticks) "
            "vs exec size in shares, from the orderbook snapshots. Bins "
            "are size octiles; alpha_loglog is the weighted log-log slope "
            "of mean |Δmid| vs mean size (instantaneous impact exponent). "
            "p_move = P(Δmid≠0 | exec). The sim's fills are all unit lots — "
            "alpha is undefined there and the unit-lot Δmid is compared "
            "to the real smallest-size bin."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
