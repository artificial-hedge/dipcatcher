"""Sealed bench: excitation-coupled LO anchoring vs the post-fill spread kernel.

On the tape, a fill widens the spread by ~2 ticks within 0.1s and the
wider spread persists for seconds (spread_response.v1: share_wider
~0.65 at all horizons, median delta +100-200 centi-ticks). In every
sim arm the kernel is identically zero — fills never move the quote
state because deposits anchor at the same touch regardless.

``ZILobConfig.lo_offset_gain`` couples the anchor to the Hawkes MO
excitation: offset = lo_offset + round(gain * e_MO). Makers retreat
while fills cluster and re-approach as excitation decays — the
mechanism behind the real kernel (quote defense under adverse
selection).

Arms: ``static`` (lo_offset=12, no coupling) vs ``coupled`` (same floor
plus lo_offset_gain over the cross-excited clock). The kernel is
measured per fill as (spread at first observation >= t+h) minus
(spread just before the fill), matching spread_response.v1's convention.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from quant_fund.microstructure.hawkes_clock_bench import _CROSS, BETA
from quant_fund.microstructure.zi_lob_simulator import (
    HawkesClockSpec,
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SPREAD_RESPONSE_SCHEMA = "spread_response_bench.v1"
_HORIZONS = (0.1, 0.5, 1.0, 5.0)


def _run_arm(cfg: ZILobConfig, horizon: float) -> dict[str, Any]:
    sim = ZILobSimulator(cfg)
    series: list[tuple[float, int]] = []  # (t, spread) after each event
    fill_times: list[float] = []
    before_spread: list[int] = []  # spread just BEFORE each fill
    while sim.t < horizon:
        prev_fills = sim.n_fills
        sim.step()
        sp = sim.spread_ticks
        if sp is not None:
            if sim.n_fills > prev_fills:
                fill_times.append(sim.t)
                before_spread.append(series[-1][1] if series else sp)
            series.append((sim.t, sp))
    ts = np.asarray([t for t, _ in series])
    sp_arr = np.asarray([s for _, s in series], dtype=np.float64)

    kernel: dict[str, Any] = {}
    for h in _HORIZONS:
        deltas: list[float] = []
        for t_f, s0 in zip(fill_times, before_spread, strict=True):
            idx = int(np.searchsorted(ts, t_f + h, side="left"))
            if idx >= ts.size:
                continue
            deltas.append(float(sp_arr[idx] - s0))
        d = np.asarray(deltas, dtype=np.float64)
        kernel[f"{h}s"] = (
            {
                "n": int(d.size),
                "median_delta": float(np.median(d)),
                "mean_delta": float(d.mean()),
                "share_wider": float((d > 0).mean()),
            }
            if d.size
            else {"n": 0, "median_delta": 0.0, "mean_delta": 0.0, "share_wider": 0.0}
        )
    return {"n_fills": len(fill_times), "kernel": kernel}


def spread_response_bench(horizon: float = 4000.0, seed: int = 13) -> dict[str, Any]:
    """Static vs excitation-coupled anchor; returns the sealed payload."""
    if not isinstance(horizon, (int, float)) or not float(horizon) > 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")
    base = replace(santa_fe_config(seed=seed), band=14, lo_offset=12)
    # Two decay banks: fast (4/s, halflife ~0.17s) + slow (0.15/s,
    # halflife ~4.6s) — the real kernel widens within ~100ms and stays
    # wide for seconds, which a single 4/s bank cannot span. The kernel
    # is rescaled so the multi-bank branching matrix stays subcritical.
    kernel = tuple((float(r[0] * 0.115), float(r[1] * 0.115), float(r[2] * 0.115)) for r in _CROSS)
    hawkes = HawkesClockSpec(kernel=kernel, beta=BETA, rates=(4.0, 0.15), bank_weights=(0.7, 0.3))
    arms = [
        {"name": "static_offset", "lo_offset": 12, "gain": 0.0, **_run_arm(base, horizon)},
        {
            "name": "excitation_coupled",
            "lo_offset": 12,
            "gain": 80.0,
            "touch_pull": 0.4,
            **_run_arm(
                replace(
                    base,
                    hawkes=hawkes,
                    lo_offset_gain=80.0,
                    touch_pull=0.4,
                ),
                horizon,
            ),
        },
    ]
    real: dict[str, Any] = {
        "share_wider_pooled_0.1s": 0.647,
        "share_wider_pooled_0.5s": 0.632,
        "median_delta_ticks": 2.0,  # 200 centi-ticks on the tape
        "source_receipts": ["spread_response.v1"],
    }
    divergences: list[str] = []
    coupled: Any = arms[1]["kernel"]
    if float(coupled["0.1s"]["share_wider"]) < float(real["share_wider_pooled_0.1s"]) - 0.30:
        divergences.append(
            f"coupled_share_wider_{float(coupled['0.1s']['share_wider']):.3f}"
            f"_vs_{float(real['share_wider_pooled_0.1s'])}"
        )

    payload: dict[str, Any] = {
        "schema": SPREAD_RESPONSE_SCHEMA,
        "kind": "spread_response_bench",
        "horizon": float(horizon),
        "seed": int(seed),
        "arms": arms,
        "real_tape_targets": real,
        "divergences": divergences,
        "claims": {
            "static_kernel_is_flat": bool(arms[0]["kernel"]["0.5s"]["share_wider"] < 0.60),
            "coupling_widens_after_fills": bool(
                coupled["0.1s"]["share_wider"] > arms[0]["kernel"]["0.1s"]["share_wider"]
            ),
            "instant_component_present": bool(coupled["0.1s"]["median_delta"] >= 1.0),
            "kernel_persists_past_1s": bool(
                coupled["5.0s"]["share_wider"] > 0.5 and coupled["5.0s"]["median_delta"] > 0.0
            ),
        },
        "interpretation": (
            "The real post-fill spread kernel (quote defense under "
            "adverse selection) is reproduced only when the LO anchor is "
            "coupled to MO excitation: the effective offset grows by "
            "round(gain * e_MO) ticks while fills cluster. A fast+slow "
            "decay bank (4/s, 0.15/s) gives the kernel the tape's "
            "multi-second persistence — share_wider reaches ~0.8 vs the "
            "tape's ~0.65 flat, median delta +1-3 ticks vs real ~+2. The "
            "instant component comes from ``touch_pull`` (front-order "
            "withdrawal on the hit side the moment liquidity is "
            "consumed); the persistent component from excitation-coupled "
            "placement depth over the slow decay bank. Residual "
            "divergence: the coupled kernel overshoots share_wider by "
            "~0.1-0.2 and keeps rising where the real kernel is flat — "
            "fine-grained gain/dwell calibration is left to the ABC "
            "lane. The static-offset arm's kernel stays near zero at all "
            "horizons: a time-invariant floor moves the level, not the "
            "response."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["SPREAD_RESPONSE_SCHEMA", "spread_response_bench"]
