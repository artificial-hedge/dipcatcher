"""flow_couple — pins and the drift kernel under ONE flow intensity.

``full_impact.v1`` measured the interference: the all-pins cell keeps
the instant channel (0.92 vs 0.887) but overshoots continuation
(8.68 vs 4.64) — because its kernel was measured under SplitFlow at the
fixed intensity 3.0 while the emptied-touch pins were closed on mixed
surfaces (the crown surface runs SplitFlow@3 internally, the reseed
surface runs iid). This bench removes the inconsistency: the SAME
config measured on BOTH surfaces AND the drift kernel at a single flow
intensity, scanned over iid / 1.0 / 2.0 / 3.0.

Measured verdict: joint_closure_found is FALSE — no uniform flow
intensity holds all seven pins and both kernel bands at once. The
mixed-surface origin of the 7/7 closure is exposed directly: under
uniform iid the spread pin fails (3.5 — the burst-driven repost/pull
machinery that re-opens the spread barely fires without metaorder
sweeps); under split flow the spread pin is seed-fragile (6.8 here vs
12.4 in full_stack.v1 at a different seed — consistent with the
joint_stability knife-edge) and k200 overshoots non-monotonically
(2.1 @ 1.0 → 9.2 @ 2.0 → 6.8 @ 3.0). The remaining residual is a
genuine damping gap on the continuation channel, not a coupling
artifact.

Evidence class: research / MIXED (sim cells vs committed tape pins).
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.crown_density_bench import _sim_crown
from quant_fund.microstructure.full_impact_bench import _FULL
from quant_fund.microstructure.full_stack_bench import _pins_ok
from quant_fund.microstructure.impact_persist_bench import _LAGS, _REAL, _measure
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.reseed_hazard_bench import sim_reseed
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

FLOW_COUPLE_SCHEMA = "flow_couple.v1"

# SplitFlow intensities; None = the default iid flow.
_INTENSITIES: tuple[float | None, ...] = (None, 1.0, 2.0, 3.0)

_INSTANT_BAND = 0.2  # |instant - 0.887|
_K200_BAND = 0.5  # |kernel[200] - 4.64|


def _kernel_ok(cell: dict[str, Any]) -> bool:
    ins = cell["instant_signed_ticks"]
    k200 = cell["kernel_mean_ticks"]["200"]
    return (
        ins is not None
        and abs(ins - _REAL["instant_signed_ticks"]) < _INSTANT_BAND
        and k200 is not None
        and abs(k200 - _REAL["kernel"]["200"]) < _K200_BAND
    )


def flow_couple_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """Scan flow intensity on the all-pins cell over all three surfaces."""
    cells: list[dict[str, Any]] = []
    for i, inten in enumerate(_INTENSITIES):
        label = "iid" if inten is None else f"split_{inten:g}"
        crown = _sim_crown(
            label,
            _FULL,
            horizon=horizon,
            seed=seed + i,
            collect_counts=True,
            flow_intensity=inten,
        )
        st = sim_reseed(label, _FULL, horizon=horizon, seed=seed + i, flow_intensity=inten)
        crown.update({k: v for k, v in st.items() if k != "regime"})
        n_fills = crown["n_fills"]
        crown["empty_share"] = round(crown["n_reveals"] / n_fills, 4) if n_fills else None
        crown["pins"] = _pins_ok(crown)
        crown["n_pins_ok"] = sum(crown["pins"].values())
        flow = _split(inten, seed + i) if inten is not None else None
        # _sim_crown's internal cfg is deterministic in (seed, extra).
        cfg = _calibrated(seed + i, _FULL)
        km = _measure(cfg, flow, horizon)
        crown["kernel_n_fills"] = km["n_fills"]
        crown["instant_signed_ticks"] = km["instant_signed_ticks"]
        crown["instant_abs_ticks"] = km["instant_abs_ticks"]
        crown["kernel_mean_ticks"] = km["kernel_mean_ticks"]
        crown["flow_intensity"] = inten
        crown["kernel_in_band"] = _kernel_ok(crown)
        crown["all_closed"] = bool(crown["kernel_in_band"] and all(crown["pins"].values()))
        cells.append(crown)

    divergences: list[str] = []
    for c in cells:
        for pin, ok in c["pins"].items():
            if not ok:
                divergences.append(f"{c['regime']}:{pin}_out")
        ins = c["instant_signed_ticks"]
        if ins is not None and abs(ins - _REAL["instant_signed_ticks"]) > _INSTANT_BAND:
            divergences.append(f"{c['regime']}:instant_{ins:+.2f}")
        k200 = c["kernel_mean_ticks"]["200"]
        if k200 is not None and abs(k200 - _REAL["kernel"]["200"]) > _K200_BAND:
            divergences.append(f"{c['regime']}:k200_{k200:+.2f}")

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    split_cells = [c for c in cells if c["flow_intensity"] is not None]
    claims = {
        "cells_measured": all(c["n_fills"] > 0 for c in cells),
        # The emptied-touch pins hold on the composed cell under the
        # iid flow the reseed surface was originally measured with.
        "pins_hold_at_iid": bool(cells[0]["n_pins_ok"] == 7),
        # ... and under persistent SplitFlow at some intensity.
        "pins_survive_flow": any(c["n_pins_ok"] == 7 for c in split_cells),
        # Capstone: one flow intensity closes every surface at once.
        "joint_closure_found": any(c["all_closed"] for c in cells),
        # Continuation scales with flow persistence — monotone k200 in
        # split intensity (nondecreasing) is the honest expectation.
        "kernel_scales_with_intensity": all(
            _f(a["kernel_mean_ticks"]["200"]) <= _f(b["kernel_mean_ticks"]["200"]) + 0.5
            for a, b in zip(split_cells, split_cells[1:], strict=False)
        ),
    }
    body: dict[str, Any] = {
        "schema": FLOW_COUPLE_SCHEMA,
        "kind": "sim_vs_real",
        "git_revision": git_revision(),
        "research_only": True,
        "data_label": "MIXED",
        "horizon": horizon,
        "seed": seed,
        "lags": list(_LAGS),
        "intensities": [("iid" if i is None else i) for i in _INTENSITIES],
        "tape_targets": _REAL,
        "cells": cells,
        "divergences": divergences,
        "claims": claims,
        "notes": (
            "Every cell is the all-pins config (joint + fill reposts + "
            "paired retreat) measured on all three surfaces at the same "
            "flow intensity: _sim_crown (crown/spread/empty/hidden/gap), "
            "sim_reseed (rate/touch), and the impact kernel "
            "(instant + lags). _sim_crown and sim_reseed each take an "
            "optional flow_intensity now; defaults (3.0 / None) preserve "
            "every historical cell bit-identically. Verdict: no uniform "
            "intensity closes all surfaces — the earlier 7/7 was a "
            "mixed-surface measure (crown under split@3, reseed under "
            "iid); under one flow the spread pin fails at every "
            "intensity tried and k200 overshoots non-monotonically "
            "under split."
        ),
    }
    body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return body
