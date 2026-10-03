"""metaorder_decay — transient-vs-permanent impact split on the ZI-LOB.

A metaorder pushed through the book lifts the mid; once the flow stops,
ZI liquidity refills around the (possibly drifting) reference level and
the mid partially reverts. The fraction retained at long horizon —
``perm_share`` — is the permanent component of impact; the rest is the
transient walk-back that real propagator models (Bacry et al.) assign
to flow that stops informing the tape.

The sim's ``ref_halflife`` knob is the mechanism: ``0`` freezes the
reference at the seed mid (strong pull-back), ``>0`` lets it track the
mid (impact becomes more permanent). This bench measures the split
under both anchors — SYNTHETIC mechanism evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    Side,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

METAORDER_DECAY_SCHEMA = "metaorder_decay.v1"

_LAGS = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class DecayResult:
    """One metaorder's post-cessation impact path (sign-normalized)."""

    peak: float  # impact at last fill vs pre-metaorder mid, in ticks
    curve: NDArray[np.float64]  # retained-impact ratio at each lag
    perm_share: float  # retained fraction at the longest lag


def _mid_ticks(sim: ZILobSimulator, base: float, tick: float) -> float | None:
    m = sim.mid
    return None if m is None else (m - base) / tick


def run_metaorder(
    *,
    config: ZILobConfig,
    side: Side,
    qty: int,
    pace_steps: int,
    warmup: int = 200,
    settle_lags: tuple[int, ...] = _LAGS,
) -> DecayResult:
    """Inject a paced metaorder, then watch the mid revert.

    ``pace_steps`` is the number of background sim steps between each
    unit-sized market order — the metaorder walks the book at a rate
    comparable to ambient flow rather than as one shock.
    """
    if qty < 1:
        raise ValueError(f"qty must be >= 1, got {qty!r}")
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    sim = ZILobSimulator(config)
    for _ in range(warmup):
        sim.step()
    for _ in range(400):
        if sim.mid is not None:
            break
        sim.step()
    base = sim.mid
    if base is None:
        raise RuntimeError("book has no mid after warmup")
    sign = 1.0 if side == "buy" else -1.0
    peak = 0.0
    for _ in range(qty):
        sim.inject_market_order(side, 1)
        for _ in range(pace_steps):
            sim.step()
        imp = _mid_ticks(sim, base, config.tick)
        if imp is not None:
            peak = max(peak, sign * imp)
    curve = np.zeros(len(settle_lags), dtype=np.float64)
    prev = 0
    for j, lag in enumerate(settle_lags):
        for _ in range(lag - prev):
            sim.step()
        prev = lag
        imp = _mid_ticks(sim, base, config.tick)
        ratio = 0.0
        if peak > 0.0 and imp is not None:
            ratio = (sign * imp) / peak
        curve[j] = ratio
    return DecayResult(peak=peak, curve=curve, perm_share=float(curve[-1]))


def fit_reversion_exponent(curve: NDArray[np.float64], lags: NDArray[np.float64]) -> float:
    """Power-law exponent of the *reversion* tail: r(ℓ) ≈ ℓ^{-γ}.

    Fit on the retained-ratio curve after its peak index; a larger γ
    means faster decay back toward the pre-metaorder mid.
    """
    mask = (curve > 0.0) & np.isfinite(curve)
    if int(mask.sum()) < 3:
        return float("nan")
    x = np.log(lags[mask])
    y = np.log(curve[mask])
    slope = float(np.polyfit(x, y, 1)[0])
    return -slope


def metaorder_decay_bench(
    *,
    n_trials: int = 12,
    qty: int = 8,
    pace_steps: int = 4,
    warmup: int = 200,
) -> dict[str, Any]:
    """Decay split under both reference anchors (frozen vs tracking)."""
    arms: dict[str, list[DecayResult]] = {
        "anchor_touch": [],
        "anchor_ref_frozen": [],
        "anchor_ref_track": [],
    }
    for k in range(n_trials):
        for arm, anchor, hl in (
            ("anchor_touch", "touch", 0.0),
            ("anchor_ref_frozen", "ref", 0.0),
            ("anchor_ref_track", "ref", 40.0),
        ):
            cfg = ZILobConfig(
                seed=5000 + k,
                anchor=anchor,
                ref_halflife=hl,
                init_depth=8,
                band=8,
            )
            arms[arm].append(
                run_metaorder(
                    config=cfg,
                    side="buy",
                    qty=qty,
                    pace_steps=pace_steps,
                    warmup=warmup,
                )
            )

    lags = np.asarray(_LAGS, dtype=np.float64)
    out_arms: dict[str, dict[str, Any]] = {}
    for name, rows in arms.items():
        peaks = np.asarray([r.peak for r in rows])
        curves = np.vstack([r.curve for r in rows])
        mean_curve = curves.mean(axis=0)
        out_arms[name] = {
            "peak_ticks_mean": float(peaks.mean()),
            "curve_mean": [float(v) for v in mean_curve],
            "perm_share_mean": float(np.mean([r.perm_share for r in rows])),
            "gamma": fit_reversion_exponent(mean_curve, lags),
        }
    frozen = out_arms["anchor_ref_frozen"]
    tracking = out_arms["anchor_ref_track"]
    touch = out_arms["anchor_touch"]
    payload: dict[str, Any] = {
        "schema": METAORDER_DECAY_SCHEMA,
        "kind": "metaorder_decay",
        "n_trials": n_trials,
        "qty": qty,
        "pace_steps": pace_steps,
        "warmup": warmup,
        "lags": list(_LAGS),
        "arms": out_arms,
        "ref_anchor_decays_more_than_touch": bool(
            frozen["perm_share_mean"] < touch["perm_share_mean"]
        ),
        "frozen_reverts_more": bool(frozen["perm_share_mean"] < tracking["perm_share_mean"]),
        "interpretation": (
            "touch anchor = the band re-forms ahead of the metaorder so "
            "impact persists (perm≈1); a frozen reference deposits liquidity "
            "in absolute price space and pulls the mid back (transient "
            "component); a tracking reference interpolates — all three arms "
            "reported verbatim"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
