"""spread_reopen — reopen the standing spread under iid via lo_offset.

``pin_stability.v1`` showed the all-pins cell's spread pin holds 0/6
under iid flow: the paired-pull/repost machinery only fires on emptied
touches, and without metaorder sweeps the touch rarely empties, so the
spread equilibrates at the placement width (~3.5 ticks) rather than
the tape's standing 9-21. ``spread_floor.v1`` already showed
``lo_offset`` floors the spread directly — but the emptied-touch stack
carries lo_offset=4 and every other pin was tuned there.

Measured verdict: spread_reopens_under_iid is FALSE. On the full
cell, deepening the anchor does not widen the standing spread —
lo8_iid lands at 4.7 and lo12_iid at 1.9 ticks (non-monotone!). The
stacked emptied-touch mechanisms dominate the placement geometry that
spread_floor.v1 exploited on the bare Santa Fe base: under the joint
knobs the equilibrium spread is set by the event-mix equilibrium, not
the anchor. lo8_iid even over-empties (0.76 vs the 0.663 band edge).
The standing-spread residual is confirmed as an equilibrium property,
not a placement offset — the next lever is the resting-density
profile itself (crown band shape), not where deposits anchor.

Evidence class: research / MIXED (sim cells vs committed tape pins).
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.crown_density_bench import _sim_crown
from quant_fund.microstructure.full_impact_bench import _FULL
from quant_fund.microstructure.full_stack_bench import _pins_ok
from quant_fund.microstructure.reseed_hazard_bench import sim_reseed
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SPREAD_REOPEN_SCHEMA = "spread_reopen.v1"

# lo_offset deltas on the all-pins cell, x flow class.
_CELLS: tuple[tuple[str, int, float | None], ...] = (
    ("lo4_iid", 4, None),
    ("lo8_iid", 8, None),
    ("lo12_iid", 12, None),
    ("lo8_split", 8, 2.0),
    ("lo12_split", 12, 2.0),
)


def spread_reopen_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    """lo_offset x flow scan on the all-pins cell."""
    cells: list[dict[str, Any]] = []
    for i, (label, lo_off, inten) in enumerate(_CELLS):
        extra = dict(_FULL, lo_offset=lo_off)
        cr = _sim_crown(
            label,
            extra,
            horizon=horizon,
            seed=seed + i,
            collect_counts=True,
            flow_intensity=inten,
        )
        st = sim_reseed(label, extra, horizon=horizon, seed=seed + i, flow_intensity=inten)
        cr.update({k: v for k, v in st.items() if k != "regime"})
        n_f = cr["n_fills"]
        cr["empty_share"] = round(cr["n_reveals"] / n_f, 4) if n_f else None
        cr["pins"] = _pins_ok(cr)
        cr["n_pins_ok"] = sum(cr["pins"].values())
        cr["lo_offset"] = lo_off
        cr["flow_intensity"] = inten
        cells.append(cr)

    divergences: list[str] = []
    for c in cells:
        for pin, ok in c["pins"].items():
            if not ok:
                divergences.append(f"{c['regime']}:{pin}_out")

    iid_cells = [c for c in cells if c["flow_intensity"] is None]
    claims = {
        "cells_measured": all(c["n_fills"] > 0 for c in cells),
        # A deeper LO anchor reopens the standing spread under iid —
        # the pin the pull/repost machinery could not carry.
        "spread_reopens_under_iid": any(
            c["pins"]["spread"] for c in iid_cells if c["lo_offset"] > 4
        ),
        # ... without destroying the emptied-touch channel it was
        # stacked on top of.
        "emptied_touch_survives": any(
            c["pins"]["spread"] and c["pins"]["empty"] for c in iid_cells
        ),
        # The deepest offset does not push the spread past the band
        # ceiling (63) — the occupancy floor is bounded. Fails closed
        # on unmeasured cells.
        "spread_bounded_above": all(
            c["spread_mean"] is not None and c["spread_mean"] < 63.0 for c in cells
        ),
    }
    body: dict[str, Any] = {
        "schema": SPREAD_REOPEN_SCHEMA,
        "kind": "sim_vs_real",
        "git_revision": git_revision(),
        "research_only": True,
        "data_label": "MIXED",
        "horizon": horizon,
        "seed": seed,
        "cells": cells,
        "divergences": divergences,
        "claims": claims,
        "notes": (
            "lo_offset shifts the standing placement anchor; the "
            "paired-pull machinery only responds to emptied touches. "
            "The hypothesis under test: under iid flow the standing "
            "spread is set by placement geometry (lo_offset), not by "
            "the post-fill response channel — so a deeper anchor should "
            "reopen the spread pin without needing sweeps. Falsified: "
            "on the full cell the anchor does not control the standing "
            "spread (lo12_iid = 1.9 ticks < lo4_iid = 2.4, non-monotone) "
            "— the stacked mechanisms dominate placement geometry; the "
            "residual is an equilibrium-density property."
        ),
    }
    body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return body
