"""band_shape — the standing spread is a deposit-density property.

``spread_reopen.v1`` falsified the placement anchor: under the full
emptied-touch stack, ``lo_offset`` does not widen the standing spread
(lo12_iid lands *narrower* than lo4_iid). The reason is structural:
with ``anchor="ref"`` and ``ref_halflife=0``, deposits land in
*absolute* price space around a frozen reference level, distributed
``P(d) ∝ d**density_exponent`` over ``d ∈ [1, band]``. The standing
spread is then the gap between the innermost *occupied* levels on each
side — a property of the density SHAPE, not the anchor offset. At
β=1 (triangular) enough mass lands at d=1 to keep the inner levels
populated and the spread tight.

This bench scans ``density_exponent`` {1, 2, 3} on the all-pins cell.
Measured verdict: shape_opens_spread is FALSE — and the response is
non-monotone a second time. b2_iid reaches 5.9 ticks and b2_split
8.45 (just under the tape band edge 9) but nothing enters [9, 63],
and b3_iid falls back to 2.3. Under the emptied-touch stack the
standing spread is robust to BOTH the anchor offset and the deposit
shape family — the touch is re-formed by the crown/iceberg/repost
machinery faster than placement thinness can hold it open. Two
falsifications locate the residual precisely: the tape's standing
9-21-tick spread is an equilibrium property this grammar cannot
express (it needs either occupied-level sparsity inside the spread
or a maker side that simply never quotes inside ~10 ticks — a
behavioral channel, not a mechanical one).

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

BAND_SHAPE_SCHEMA = "band_shape.v1"

# density_exponent x flow on the all-pins cell.
_CELLS: tuple[tuple[str, float, float | None], ...] = (
    ("b1_iid", 1.0, None),
    ("b2_iid", 2.0, None),
    ("b3_iid", 3.0, None),
    ("b2_split", 2.0, 2.0),
    ("b3_split", 3.0, 2.0),
)


def band_shape_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    """density_exponent x flow scan on the all-pins cell."""
    cells: list[dict[str, Any]] = []
    for i, (label, beta, inten) in enumerate(_CELLS):
        extra = dict(_FULL, density_exponent=beta)
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
        cr["density_exponent"] = beta
        cr["flow_intensity"] = inten
        cells.append(cr)

    divergences: list[str] = []
    for c in cells:
        for pin, ok in c["pins"].items():
            if not ok:
                divergences.append(f"{c['regime']}:{pin}_out")

    iid_cells = [c for c in cells if c["flow_intensity"] is None]
    steep = [c for c in iid_cells if c["density_exponent"] > 1.0]
    claims = {
        "cells_measured": all(c["n_fills"] > 0 for c in cells),
        # Steepening the deposit density opens the standing spread under
        # iid — the pin no post-fill mechanism could carry.
        "shape_opens_spread": any(c["pins"]["spread"] for c in steep),
        # The deposit shape spreads monotone-ish: every steeper cell is
        # wider than the triangular base.
        "spread_monotone_in_beta": all(
            (c["spread_mean"] or 0.0) > (iid_cells[0]["spread_mean"] or 0.0) for c in steep
        ),
        # The emptied-touch pins survive the density reshaping.
        "emptied_touch_survives": any(c["pins"]["spread"] and c["pins"]["empty"] for c in steep),
    }
    body: dict[str, Any] = {
        "schema": BAND_SHAPE_SCHEMA,
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
            "With anchor='ref' at ref_halflife=0, deposits distribute "
            "P(d) ∝ d^beta over [1, band] around a frozen reference — "
            "the standing spread is the innermost-occupied gap, a "
            "density-shape property. beta=1 (triangular) keeps the "
            "inner levels populated; beta>1 pushes mass to the outer "
            "band edge. This tests whether the residual standing-spread "
            "gap is reachable through the placement law itself. "
            "Falsified: no beta enters the tape band (b2_split peaks at "
            "8.45 vs band edge 9); the touch re-forms via the "
            "crown/iceberg/repost machinery faster than thin placement "
            "holds it open — the residual is an equilibrium property "
            "this event grammar cannot express."
        ),
    }
    body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return body
