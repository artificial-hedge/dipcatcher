"""zone_card — does the closed cell also carry the tape's event grammar?

``zone_embargo.v1``/``zone_stability.v1`` closed the standing spread:
z12-iid holds all seven pins at rate 0.75, occupancy 0.96 vs 0.80,
and the drift kernel lands near tape. But the tape's grammar is more
than the spread: its event card is 48.9% submissions / 45.8% deletes /
3.3% execs (event_matrix.v1), and its resting orders are fast —
executed makers live a median 2.21s ~= 25.5 events at the tape's
11.53 ev/s (order_lifetime.v1). This bench measures the closed cells'
event-mix shares and per-fill maker ages to check whether the zone
also reproduces the grammar, or only the geometry.

Receipts are sealed via ``receipt_sha256`` and carry ``data_label``
``"MIXED"`` — synthetic draws measured against committed real-tape
event-card and lifetime pins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.full_impact_bench import _FULL
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

ZONE_CARD_SCHEMA = "zone_card.v1"

# Committed tape card (event_matrix.v1: mix_share over 269,748 events)
# and resting-order lifetime pins (order_lifetime.v1, seconds; the
# tape runs 269748 events / 23399.9 s = 11.528 ev/s).
_TAPE_MIX = {"sub": 0.4892, "delete": 0.4577, "exec": 0.0333}
_TAPE_EV_PER_S = 11.528
_TAPE_LIFE_S = {"executed_p50": 2.213, "deleted_p50": 0.800, "p90_all": 13.671}

# (label, zone_embargo, flow_intensity) — the closure corner plus its
# neighbors and the zone-free baseline.
_CELLS: tuple[tuple[str, int, float | None], ...] = (
    ("z0_iid", 0, None),
    ("z0_split", 0, 2.0),
    ("z10_iid", 10, None),
    ("z12_iid", 12, None),
    ("z12_split", 12, 2.0),
    ("z14_iid", 14, None),
)

_SEEDS = (7, 11)


def _zone_cell(zone: int, inten: float | None, *, horizon: int, seed: int) -> dict[str, Any]:
    extra = dict(
        _FULL,
        zone_embargo=zone,
        fill_repost_frac=0.8,
        fill_repost_delay=280,
        repost_frac=0.6,
        repost_band=4,
        repost_window=500,
    )
    cfg = _calibrated(seed, extra)
    flow = _split(inten, seed + 1) if inten is not None else None
    sim = ZILobSimulator(cfg, flow)
    ages: list[float] = []
    seen = 0
    for _ in range(horizon):
        sim.step()
        while seen < len(sim.trades):
            tr = sim.trades[seen]
            seen += 1
            if tr.maker_t_submit is not None:
                ages.append(sim._t - float(tr.maker_t_submit))
    counts = sim.event_counts()
    subs = float(counts.get("n_lo_units", 0))
    dels = float(counts.get("n_cancellations", 0))
    execs = float(sim.n_fills)
    tot = subs + dels + execs
    ages_a = np.asarray(ages, dtype=float)
    n_ages = len(ages_a)
    # Maker ages are measured on the sim's own clock (self._t is
    # seconds); convert to event units via the draw's realized event
    # rate so the tape's per-event lifetime compares honestly.
    rate_ev_per_t = sim.n_events / sim._t if sim._t > 0 else 0.0
    ages_ev = ages_a * rate_ev_per_t
    life_ev_p50 = float(np.median(ages_ev)) if n_ages else None
    life_ev_p90 = float(np.quantile(ages_ev, 0.9)) if n_ages else None
    return {
        "n_events": sim.n_events,
        "n_fills": sim.n_fills,
        "mix_share": {
            "sub": round(subs / tot, 4) if tot else None,
            "delete": round(dels / tot, 4) if tot else None,
            "exec": round(execs / tot, 4) if tot else None,
        },
        "life_events_p50_executed": life_ev_p50,
        "life_events_p90_all": life_ev_p90,
        # Tape-equivalent seconds: events at the tape's own event rate.
        "life_tape_seconds_p50_executed": (
            round(life_ev_p50 / _TAPE_EV_PER_S, 3) if life_ev_p50 is not None else None
        ),
        "counts": {k: counts[k] for k in ("n_lo_arrivals", "n_mo_arrivals", "n_cancellations")},
    }


def zone_card_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for label, zone, inten in _CELLS:
        draws = [_zone_cell(zone, inten, horizon=horizon, seed=seed * 1000 + s) for s in _SEEDS]
        sub = [d["mix_share"]["sub"] for d in draws if d["mix_share"]["sub"] is not None]
        dele = [d["mix_share"]["delete"] for d in draws if d["mix_share"]["delete"] is not None]
        exc = [d["mix_share"]["exec"] for d in draws if d["mix_share"]["exec"] is not None]
        life = [
            d["life_events_p50_executed"]
            for d in draws
            if d["life_events_p50_executed"] is not None
        ]
        cells.append(
            {
                "regime": label,
                "zone_embargo": zone,
                "flow": "iid" if inten is None else "split",
                "n_draws": len(draws),
                "mix_mean": {
                    "sub": round(sum(sub) / len(sub), 4) if sub else None,
                    "delete": round(sum(dele) / len(dele), 4) if dele else None,
                    "exec": round(sum(exc) / len(exc), 4) if exc else None,
                },
                "life_ev_p50_mean": round(sum(life) / len(life), 2) if life else None,
                "draws": draws,
            }
        )

    def _tape_gap(c: dict[str, Any]) -> float | None:
        """Mean |share - tape| across the three headline classes."""
        m = c["mix_mean"]
        if None in (m["sub"], m["delete"], m["exec"]):
            return None
        return float(
            np.mean(
                [
                    abs(m["sub"] - _TAPE_MIX["sub"]),
                    abs(m["delete"] - _TAPE_MIX["delete"]),
                    abs(m["exec"] - _TAPE_MIX["exec"]),
                ]
            )
        )

    for c in cells:
        gap = _tape_gap(c)
        c["tape_mix_gap"] = round(gap, 4) if gap is not None else None

    # Tape's executed-maker p50 in event units.
    tape_life_ev = _TAPE_LIFE_S["executed_p50"] * _TAPE_EV_PER_S
    claims = {
        "cells_measured": all(c["n_draws"] == len(_SEEDS) for c in cells),
        # The zone cells' mix is closer to the tape card than the
        # zone-free baseline under the same flow.
        "zone_improves_card": any(
            (c["tape_mix_gap"] or 1.0) < (cells[0]["tape_mix_gap"] or 1.0)
            for c in cells
            if c["zone_embargo"] > 0 and c["flow"] == "iid"
        ),
        # Executed-maker lifetime reaches the tape's ~25.5-event median
        # (within 2x) on at least one cell.
        "life_on_tape_scale": any(
            c["life_ev_p50_mean"] is not None
            and 0.5 * tape_life_ev <= c["life_ev_p50_mean"] <= 2.0 * tape_life_ev
            for c in cells
        ),
        # Exec share near the tape's 3.3% on at least one cell.
        "exec_share_on_card": any(
            c["mix_mean"]["exec"] is not None
            and abs(c["mix_mean"]["exec"] - _TAPE_MIX["exec"]) <= 0.03
            for c in cells
        ),
    }

    out: dict[str, Any] = {
        "schema": ZONE_CARD_SCHEMA,
        "kind": "microstructure_bench",
        "research_only": True,
        "data_label": "MIXED",
        "git_revision": git_revision(),
        "horizon": horizon,
        "seed": seed,
        "config": {
            "seeds_per_cell": list(_SEEDS),
            "cells": [
                {"label": label, "zone_embargo": z, "flow_intensity": i} for label, z, i in _CELLS
            ],
        },
        "tape_reference": {
            "mix_share": _TAPE_MIX,
            "events_per_s": _TAPE_EV_PER_S,
            "life_seconds": _TAPE_LIFE_S,
            "executed_p50_events": round(tape_life_ev, 1),
            "sources": ["event_matrix.v1", "order_lifetime.v1"],
        },
        "cells": cells,
        "claims": claims,
        "notes": (
            "Event grammar on the closed cells: mix shares over "
            "(submissions, deletes, execs) from sim.event_counts and "
            "per-fill maker age in events (tr.maker_t_submit), "
            "converted to tape-equivalent seconds at the tape's own "
            "11.528 ev/s. The zone closes geometry (spread/occupancy/"
            "kernel); this measures whether the closed cell's event "
            "card and order lifetime also sit on the tape's values — "
            "48.9%/45.8%/3.3% mix, 25.5-event executed p50. "
            "Measured at 3k preview + 15k: geometry closes but "
            "grammar does not — executed-maker p50 sits ~90-250 "
            "events vs the tape's ~25.5 (~4-10x), and the zone "
            "cells' mix gap is slightly WORSE than baseline "
            "(0.047-0.056 vs 0.041): the no-quote zone adds depth "
            "at the edge, inflating sub share. The residual is now "
            "cleanly an event-rate channel: the tape churns ~14 "
            "deletes per fill at ~26-event maker horizons; the "
            "sim's makers rest far too long."
        ),
    }
    body = dict(out)
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="receipts/zone_card.json")
    args = ap.parse_args()
    out = zone_card_bench(horizon=args.horizon, seed=args.seed)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
