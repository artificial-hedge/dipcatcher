"""full_impact — does the all-pins cell also hold the impact channels?

``full_stack.v1`` closed the emptied-touch channel: ``joint_evt80_pp60_b4``
holds all seven tape pins at once. The separate wave-23 question was the
drift kernel — instant impact ~0.89 ticks and continuation ~4.64 ticks at
+200 events (impact_persist.v1). Those were closed in different cells too:
vacancy memory (refill_cooldown) produces the instant term, SplitFlow the
continuation. This bench asks whether the emptied-touch composition
*interferes* with the drift channels — the paired retreat and re-seeding
directly reshape the post-fill book.

Cells: deep baseline, the joint cell, the all-pins cell, each with and
without SplitFlow (the tape's continuation needs persistent flow), plus a
SplitFlow variant of the pinned winner. Divergences use the source bench's
tolerances (|kernel - tape| > 0.5 ticks per lag, |instant - 0.887| > 0.2).

Measured verdict: the instant channel SURVIVES the composition — the
all-pins cell prints 0.81 solo / 0.92 under SplitFlow vs the tape's
0.887 — but the continuation overshoots: 6.85 ticks at +200 vs 4.64
(joint+split already 6.19). The emptied-touch machinery amplifies
SplitFlow's drift; the residual is now a *damping* problem on the
continuation channel, not a missing mechanism.

Evidence class: research / SYNTHETIC (sim cells vs committed tape pins).
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.impact_persist_bench import _LAGS, _REAL, _measure
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.reseed_hazard_bench import _JOINT
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

FULL_IMPACT_SCHEMA = "full_impact.v1"

# The all-pins cell from full_stack.v1.
_FULL = dict(
    _JOINT,
    fill_repost_frac=0.8,
    fill_repost_delay=160,
    paired_pull_frac=0.6,
    paired_pull_band=4,
)

# (label, extra deltas on the calibrated base, use SplitFlow?)
_CELLS: tuple[tuple[str, dict[str, Any], bool], ...] = (
    ("deep", dict(), False),
    ("joint", dict(_JOINT), False),
    ("joint", dict(_JOINT), True),
    ("full", dict(_FULL), False),
    ("full", dict(_FULL), True),
)


def full_impact_bench(*, horizon: int = 60000, seed: int = 7) -> dict[str, Any]:
    """Drift-kernel pins on the emptied-touch composition cells."""
    cells: list[dict[str, Any]] = []
    for i, (label, extra, split) in enumerate(_CELLS):
        cfg: ZILobConfig = _calibrated(seed + i, extra)
        flow = _split(3.0, seed + i) if split else None
        m = _measure(cfg, flow, horizon)
        m["regime"] = f"{label}{'_split' if split else ''}"
        cells.append(m)

    divergences: list[str] = []
    for c in cells:
        ins = c["instant_signed_ticks"]
        if ins is not None and abs(ins - _REAL["instant_signed_ticks"]) > 0.2:
            divergences.append(
                f"{c['regime']}:instant_{ins:+.2f}_vs_{_REAL['instant_signed_ticks']:.2f}"
            )
        for lag in _LAGS:
            got = c["kernel_mean_ticks"][str(lag)]
            want = _REAL["kernel"][str(lag)]
            if got is None:
                divergences.append(f"{c['regime']}@{lag}:no_fills")
            elif abs(got - want) > 0.5:
                divergences.append(f"{c['regime']}@{lag}:{got:+.2f}_vs_{want:.2f}")

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    full_split = cells[4]
    claims = {
        "cells_measured": all(c["n_fills"] > 0 for c in cells),
        # The all-pins cell under persistent flow lands instant in ±0.2
        # and the +200 continuation within ±0.5 of the tape.
        "full_stack_carries_instant": abs(
            _f(full_split["instant_signed_ticks"]) - _REAL["instant_signed_ticks"]
        )
        < 0.2,
        "full_stack_carries_continuation": abs(
            _f(full_split["kernel_mean_ticks"]["200"]) - _REAL["kernel"]["200"]
        )
        < 0.5,
        # The paired retreat should not erase the continuation channel.
        "repost_preserves_kernel": bool(
            _f(full_split["kernel_mean_ticks"]["200"])
            > _f(cells[1]["kernel_mean_ticks"]["200"]) - 1.0
        ),
    }
    body: dict[str, Any] = {
        "schema": FULL_IMPACT_SCHEMA,
        "kind": "sim_vs_real",
        "git_revision": git_revision(),
        "research_only": True,
        "data_label": "MIXED",
        "horizon": horizon,
        "seed": seed,
        "lags": list(_LAGS),
        "tape_targets": _REAL,
        "cells": cells,
        "divergences": divergences,
        "claims": claims,
        "notes": (
            "Cross-channel interference audit: the all-pins emptied-touch "
            "cell (joint + fill reposts + paired retreat) is re-run through "
            "the drift-kernel measure with and without SplitFlow. The "
            "question is whether closing the vacancy channel broke the "
            "impact channels — fill reposts refill the touch the sweep "
            "displaced (attenuating instant), and the paired retreat "
            "widens the spread the kernel is measured across. Measured: "
            "instant survives (0.92 vs 0.887 under split flow), "
            "continuation overshoots (6.85 vs 4.64 at +200) — the "
            "residual is damping the amplified drift, not a missing "
            "mechanism."
        ),
    }
    body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return body
