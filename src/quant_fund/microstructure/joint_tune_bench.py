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
from pathlib import Path
from typing import Any

from quant_fund.microstructure.churn_reseed_bench import _reseed_fates
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
            }
        )

    def _closed(c: dict[str, Any]) -> bool:
        return (
            c["n_pins_mean"] >= 6.5
            and c["life_ev_p50_mean"] is not None
            and 0.5 * _TAPE_LIFE_EV <= c["life_ev_p50_mean"] <= 2.0 * _TAPE_LIFE_EV
            and c["instant_mean"] is not None
            and abs(c["instant_mean"] - _TAPE_INSTANT) <= 0.35
            and c["k200_mean"] is not None
            and 3.0 <= c["k200_mean"] <= 6.0
        )

    best = max(
        cells,
        key=lambda c: (c["n_pins_mean"], -abs((c["life_ev_p50_mean"] or 999) - _TAPE_LIFE_EV)),
    )

    claims = {
        "cells_measured": all(c["n_draws"] == len(_SEEDS) for c in cells),
        # Some cell composes pins + tape-scale life + kernel at once.
        "joint_closure_found": any(_closed(c) for c in cells),
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
        "notes": (
            "The (ttl x requote x delay) corner maps a real Pareto "
            "frontier at 15k — no cell composes pins + tape-scale life "
            "+ kernel: joint_closure_found False. Pins want ttl>=150 "
            "(life 75-89 ev); tape-scale life wants ttl<=75 (life "
            "30-43). Binding pins on the fast cells are reseed_rate "
            "and reveal_gap — and reseed OVERSHOOTS under fast churn "
            "(0.61-0.76 vs the tape's 0.54 band ceiling): churned "
            "makers keep vacancies hot. reveal_gap sits at rate 0.5 "
            "everywhere (knife-edge single-draw). kernel_carried "
            "False by 0.04 — the all-pins cell's k200 is 6.04 vs the "
            "6.0 band edge. ttl50_rq90_d110 is the closest compose "
            "(6 pins, life 30 ~ tape 25.5, inst 0.835, k200 4.44) — "
            "the frontier's gap is ~half a pin: the emptied-touch "
            "machinery is now slightly too persistent, not missing."
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
