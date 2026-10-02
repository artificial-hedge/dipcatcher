"""joint_tune — is there a cell holding pins + grammar + kernel together?

``repost_latency.v1`` produced two adjacent near-closures:
``joint_d60`` (all 7 pins on both draws, instant in tolerance, but
life 89 ev vs the tape's 25.5 and k200 6.04 vs 4.64) and
``fast_d110`` (life 30 ev ~ tape, k200 4.44, instant 0.835, 6 pins).
This bench sweeps the (ttl x requote x delay) corner around them to
test whether one configuration composes all four channels at once —
pins + tape-scale maker life + drift kernel — or whether the
grammar/kernel trade-off is a real frontier.

Receipts are sealed via ``receipt_sha256`` and carry ``data_label``
``"MIXED"`` — synthetic draws measured against committed real-tape
pins.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from quant_fund.microstructure.churn_reseed_bench import _reseed_fates
from quant_fund.microstructure.full_stack_bench import _PIN_FIELDS
from quant_fund.microstructure.zone_card_bench import _TAPE_EV_PER_S, _TAPE_LIFE_S
from quant_fund.microstructure.zone_ttl_bench import _cell
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

JOINT_TUNE_SCHEMA = "joint_tune.v1"

# (label, zone, ttl, requote, fill_repost_delay) — iid flow throughout;
# split flow broke the emptied-touch pin rate on every prior cell.
_CELLS: tuple[tuple[str, int, int, float, int], ...] = (
    ("ttl50_rq90_d110", 12, 50, 0.9, 110),
    ("ttl50_rq90_d60", 12, 50, 0.9, 60),
    ("ttl75_rq90_d60", 12, 75, 0.9, 60),
    ("ttl75_rq90_d110", 12, 75, 0.9, 110),
    ("ttl100_rq75_d60", 12, 100, 0.75, 60),
    ("ttl100_rq75_d110", 12, 100, 0.75, 110),
    ("ttl150_rq75_d60", 12, 150, 0.75, 60),
    ("ttl200_rq60_d60", 12, 200, 0.6, 60),
)

_SEEDS = (7, 11)

_TAPE_LIFE_EV = _TAPE_LIFE_S["executed_p50"] * _TAPE_EV_PER_S
_TAPE_K200 = 4.64
_TAPE_INSTANT = 0.887


def _draw_closed(draw: dict[str, Any]) -> bool:
    """All seven pins and all three unrounded measurements on one draw.

    A missing/nonfinite measurement fails closed. Pin counts and cell means
    are diagnostics, not substitutes for simultaneous per-draw evidence.
    """
    pins = draw["pins"]
    life = draw["card"]["life_events_p50_executed"]
    instant = draw["instant_signed_ticks"]
    k200 = draw["k200"]
    return (
        set(pins) == {pin for pin, _ in _PIN_FIELDS}
        and all(value is True for value in pins.values())
        and life is not None
        and math.isfinite(life)
        and 0.5 * _TAPE_LIFE_EV <= life <= 2.0 * _TAPE_LIFE_EV
        and instant is not None
        and math.isfinite(instant)
        and _TAPE_INSTANT - 0.35 <= instant <= _TAPE_INSTANT + 0.35
        and k200 is not None
        and math.isfinite(k200)
        and 3.0 <= k200 <= 6.0
    )


def joint_tune_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for label, zone, ttl, rq, fd in _CELLS:
        draws = [
            _cell(
                zone,
                ttl,
                None,
                horizon=horizon,
                seed=seed * 1000 + s,
                requote=rq,
                fr_delay=fd,
            )
            for s in _SEEDS
        ]
        fates = [
            _reseed_fates(
                zone,
                ttl,
                rq,
                None,
                horizon=horizon,
                seed=seed * 1000 + s,
                fr_delay=fd,
            )
            for s in _SEEDS
        ]

        def _m(key: str, src: list[dict[str, Any]]) -> float | None:
            vs = [d[key] for d in src if d[key] is not None]
            return round(sum(vs) / len(vs), 4) if vs else None

        pin_names = list(draws[0]["pins"])
        cells.append(
            {
                "regime": label,
                "zone_embargo": zone,
                "maker_ttl": ttl,
                "maker_requote": rq,
                "fill_repost_delay": fd,
                "n_draws": len(draws),
                "n_pins_mean": round(sum(d["n_pins"] for d in draws) / len(draws), 2),
                "pin_rates": {
                    p: sum(1 for d in draws if d["pins"][p]) / len(draws) for p in pin_names
                },
                "life_ev_p50_mean": _m("life_events_p50_executed", [d["card"] for d in draws]),
                "instant_mean": _m("instant_signed_ticks", draws),
                "k200_mean": _m("k200", draws),
                "reseed_rate_500": _m("reseed_rate_500", fates),
                "reseed_as_touch_share": _m("reseed_as_touch_share", fates),
                "reseed_latency_p50": _m("reseed_latency_p50", fates),
                "draws": draws,
                "joint_closure_by_draw": [_draw_closed(d) for d in draws],
            }
        )

    def _aggregate_closed(c: dict[str, Any]) -> bool:
        return (
            c["n_pins_mean"] >= 6.5
            and c["life_ev_p50_mean"] is not None
            and 0.5 * _TAPE_LIFE_EV <= c["life_ev_p50_mean"] <= 2.0 * _TAPE_LIFE_EV
            and c["instant_mean"] is not None
            and abs(c["instant_mean"] - _TAPE_INSTANT) <= 0.35
            and c["k200_mean"] is not None
            and 3.0 <= c["k200_mean"] <= 6.0
        )

    for c in cells:
        # Keep the former mean-based screen, explicitly labeled as aggregate.
        c["aggregate_closure"] = _aggregate_closed(c)
        c["joint_closure_rate"] = sum(c["joint_closure_by_draw"]) / c["n_draws"]

    best = max(
        cells,
        key=lambda c: (c["n_pins_mean"], -abs((c["life_ev_p50_mean"] or 999) - _TAPE_LIFE_EV)),
    )

    claims = {
        "cells_measured": all(c["n_draws"] == len(_SEEDS) for c in cells),
        # One configuration must compose every channel on every sampled draw.
        "joint_closure_found": any(
            bool(c["joint_closure_by_draw"]) and all(c["joint_closure_by_draw"]) for c in cells
        ),
        # Tape-scale grammar cells keep >= 6 pins mean.
        "grammar_keeps_pins": any(
            c["life_ev_p50_mean"] is not None
            and c["life_ev_p50_mean"] <= 2.0 * _TAPE_LIFE_EV
            and c["n_pins_mean"] >= 6.0
            for c in cells
        ),
        # The kernel stays in tolerance on the best-pins cell.
        "kernel_carried": (
            best["instant_mean"] is not None
            and abs(best["instant_mean"] - _TAPE_INSTANT) <= 0.35
            and best["k200_mean"] is not None
            and 3.0 <= best["k200_mean"] <= 6.0
        ),
    }

    out: dict[str, Any] = {
        "schema": JOINT_TUNE_SCHEMA,
        "kind": "microstructure_bench",
        "research_only": True,
        "data_label": "MIXED",
        "git_revision": git_revision(),
        "horizon": horizon,
        "seed": seed,
        "config": {
            "seeds": list(_SEEDS),
            "flow": "iid",
            "cells": [
                {
                    "label": label,
                    "zone_embargo": z,
                    "maker_ttl": t,
                    "maker_requote": r,
                    "fill_repost_delay": d,
                }
                for label, z, t, r, d in _CELLS
            ],
        },
        "tape_reference": {
            "life_p50_events": round(_TAPE_LIFE_EV, 1),
            "k200_ticks": _TAPE_K200,
            "instant_ticks": _TAPE_INSTANT,
            "sources": ["order_lifetime.v1", "impact_persist.v1"],
        },
        "cells": cells,
        "claims": claims,
        "claim_semantics": {
            "joint_closure_found": "all_draws.v1: all seven pins plus life, instant, and k200 "
            "within tolerance on every draw of at least one cell; uses unrounded values",
            "grammar_keeps_pins": "aggregate: mean life <= 2x tape and mean pin count >= 6",
            "kernel_carried": "aggregate: mean instant and k200 in tolerance on best-pins cell",
            "aggregate_closure": "legacy diagnostic: mean pins >= 6.5 plus mean life, instant, "
            "and k200 in tolerance; does not establish per-draw joint closure",
        },
        "notes": (
            "Synthetic draws compared with committed tape references; MIXED research-only "
            "evidence. joint_closure_found requires simultaneous per-draw closure on every "
            "sampled draw of one configuration. Cell means and aggregate_closure are "
            "descriptive diagnostics and can pass when no draw closes. A finite sweep "
            "does not establish a general Pareto frontier or market validation."
        ),
    }
    body = dict(out)
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="receipts/joint_tune.json")
    args = ap.parse_args()
    out = joint_tune_bench(horizon=args.horizon, seed=args.seed)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
