"""quote_floor — the standing spread as a maker-behavior floor.

``spread_reopen.v1`` falsified the placement anchor (``lo_offset``),
``band_shape.v1`` falsified the deposit-density family
(``density_exponent``) — no mechanical placement family opens the
tape's standing 9-21-tick spread because the emptied-touch machinery
(crown stack, fill reposts, iceberg reloads) re-forms the touch faster
than placement thinness holds it open. The remaining hypothesis is the
behavioral one: real makers simply do not quote inside a minimum depth
band — adverse-selection-aware participants keep resting flow at least
``min_quote_dist`` ticks from the reference, and interior vacancies
stay dead.

This bench scans ``min_quote_dist`` {0, 4, 8, 12} on the all-pins cell
under iid and SplitFlow@2: the floor binds the ambient density draw
(mass piles at the boundary level) and gates the repost/refill channels
(an in-band vacancy is never re-seeded), while the touch-forming
classes — crown stacks, joins, post-fill chase — still operate at
whatever touch the floor establishes. A floor that works should hold
the spread near ``2 * min_quote_dist`` while preserving the
emptied-touch pins the campaign was built for.

Measured verdict: floor_opens_spread is TRUE — g8 under SplitFlow@2
lands a standing spread of 23.8 ticks, the first cell in the
wave-23/24 campaign to enter the tape's [9, 63] band. But
floor_composes is FALSE: the same cell overshoots emptied share
(0.76 vs band edge 0.663) and drops the reseed pins, and under iid
flow the response is non-monotone (g4 opens to 8.6, g8 collapses to
1.6). The floor is a real channel — the first to move the standing
spread into the band at all — but its composition with the
emptied-touch machinery is the next interference problem.

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

QUOTE_FLOOR_SCHEMA = "quote_floor.v1"

# min_quote_dist x flow on the all-pins cell.
_CELLS: tuple[tuple[str, int, float | None], ...] = (
    ("g0_iid", 0, None),
    ("g4_iid", 4, None),
    ("g8_iid", 8, None),
    ("g12_iid", 12, None),
    ("g4_split", 4, 2.0),
    ("g8_split", 8, 2.0),
)


def quote_floor_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    """min_quote_dist x flow scan on the all-pins cell."""
    cells: list[dict[str, Any]] = []
    for i, (label, floor, inten) in enumerate(_CELLS):
        extra = dict(_FULL, min_quote_dist=floor)
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
        cr["min_quote_dist"] = floor
        cr["flow_intensity"] = inten
        cells.append(cr)

    divergences: list[str] = []
    for c in cells:
        for pin, ok in c["pins"].items():
            if not ok:
                divergences.append(f"{c['regime']}:{pin}_out")

    floored = [c for c in cells if c["min_quote_dist"] > 0]

    # Within-flow control only: a floored cell's spread demonstrates the
    # floor's effect against an unfloored cell on the SAME flow arm —
    # comparing a split-flow floor to the iid control confounds the
    # floor with the flow regime.
    def _control(c: dict[str, Any]) -> dict[str, Any] | None:
        for o in cells:
            if o["min_quote_dist"] == 0 and o["flow_intensity"] == c["flow_intensity"]:
                return o
        return None

    scale_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for c in floored:
        base = _control(c)
        if base is not None:
            scale_pairs.append((c, base))
    claims = {
        "cells_measured": all(c["n_fills"] > 0 for c in cells),
        # A maker floor opens the standing spread into the tape band.
        "floor_opens_spread": any(c["pins"]["spread"] for c in floored),
        # The floor sets the spread scale: floored cells widen over
        # their own arm's unfloored control.
        "spread_scales_with_floor": bool(scale_pairs)
        and all((c["spread_mean"] or 0.0) > (b["spread_mean"] or 0.0) for c, b in scale_pairs),
        # The emptied-touch pins survive under some floored cell — the
        # floor composes with the campaign's machinery rather than
        # breaking it.
        "floor_composes": any(
            c["pins"]["spread"] and c["pins"]["empty"] and c["pins"]["reveal_gap"] for c in floored
        ),
    }
    body: dict[str, Any] = {
        "schema": QUOTE_FLOOR_SCHEMA,
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
            "min_quote_dist floors the ambient density draw at g ticks "
            "from the ref anchor and gates the repost/fill-repost "
            "channels: a vacancy inside the no-quote band stays dead. "
            "Touch-forming classes (crown, join, improve, chase) are "
            "exempt — they operate at whatever touch the floor "
            "establishes. This is the behavioral channel the tape's "
            "standing spread needs: makers that never quote inside a "
            "minimum depth band, not a placement-law shape. Measured: "
            "g8 under SplitFlow@2 enters the tape band (23.8 ticks — "
            "the campaign's first spread-pin hit), while empty "
            "overshoots (0.76 vs 0.663 edge) and the iid cells respond "
            "non-monotonically — the floor opens the spread but does "
            "not yet compose with emptied-touch."
        ),
    }
    body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return body
