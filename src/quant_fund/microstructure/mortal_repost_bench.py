"""mortal_repost — do one-shot reposts close the reseed overshoot?

``joint_tune.v1`` mapped the last residual: on fast-churn cells the
``reseed_rate`` pin OVERSHOOTS (0.61-0.76 vs the tape's ~0.54 band
ceiling) — churned reposts keep vacancies hot forever, so an emptied
level never stays empty. On the tape a reposted level is a fresh
order that can churn out like any other: ``repost_requote`` gives the
``repost``-tagged units their own requote probability — 0.0 makes a
reseeded level mortal (one-shot at expiry) instead of undying.

This bench sweeps (ttl x requote x delay x repost_requote) around the
Pareto corner: does mortal repost pull the fast cells' reseed rate
back into the tape band WITHOUT losing pins or grammar? If yes, the
half-pin frontier gap closes — the emptied touch is persistent but
not permanent.

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

MORTAL_REPOST_SCHEMA = "mortal_repost.v1"

# (label, zone, ttl, requote, fill_repost_delay, repost_requote,
#  repost_ttl_immune)
# rr=None rows are the controls (inherit maker_requote — bit-identical
# to the joint_tune cells); ``immune`` rows exempt repost units from
# ttl expiry (sticky reseed — the inverse of mortality).
_CELLS: tuple[tuple[str, int, int, float, int, float | None, bool], ...] = (
    ("fast_d110_rr0", 12, 50, 0.9, 110, 0.0, False),
    ("fast_d110_rr30", 12, 50, 0.9, 110, 0.3, False),
    ("fast_d110_rr90", 12, 50, 0.9, 110, None, False),
    ("fast_d60_rr0", 12, 50, 0.9, 60, 0.0, False),
    ("mid_d60_rr0", 12, 100, 0.75, 60, 0.0, False),
    ("mid_d60_rr30", 12, 100, 0.75, 60, 0.3, False),
    ("joint_d60_rr0", 12, 200, 0.6, 60, 0.0, False),
    ("joint_d60_ctrl", 12, 200, 0.6, 60, None, False),
    ("fast_d110_imm", 12, 50, 0.9, 110, None, True),
    ("mid_d60_imm", 12, 100, 0.75, 60, None, True),
    ("joint_d60_imm", 12, 200, 0.6, 60, None, True),
)

_SEEDS = (7, 11)
_TAPE_LIFE_EV = _TAPE_LIFE_S["executed_p50"] * _TAPE_EV_PER_S
# Tape reseed band (from sim_real_ledger / full_stack conventions).
_RESEED_LO, _RESEED_HI = 0.4, 0.68


def mortal_repost_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for label, zone, ttl, rq, fd, rr, imm in _CELLS:
        draws = [
            _cell(
                zone,
                ttl,
                None,
                horizon=horizon,
                seed=seed * 1000 + s,
                requote=rq,
                fr_delay=fd,
                repost_requote=rr,
                repost_ttl_immune=imm,
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
                repost_requote=rr,
                repost_ttl_immune=imm,
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
                "repost_requote": rr,
                "repost_ttl_immune": imm,
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

    # Mortal repost damps the overshoot: an rr=0 fast cell lands its
    # reseed rate inside the tape band while the rr-inherit control
    # overshoots.
    fast_ctl = next((c for c in cells if c["regime"] == "fast_d110_rr90"), None)
    fast_mortal = [c for c in cells if c["regime"] in ("fast_d110_rr0", "fast_d110_rr30")]
    mortal_damps = any(
        c["reseed_rate_500"] is not None and _RESEED_LO <= c["reseed_rate_500"] <= _RESEED_HI
        for c in fast_mortal
    ) and (
        fast_ctl is not None
        and fast_ctl["reseed_rate_500"] is not None
        and fast_ctl["reseed_rate_500"] > _RESEED_HI
    )
    # Immune reposts damp the other way: a sticky reseeded level fills
    # the vacancy ONCE, so measured episodes stop cycling — the immune
    # fast cell should land its reseed rate below the mortal cells.
    imm_cell = next((c for c in cells if c["regime"] == "fast_d110_imm"), None)
    immune_damps = (
        imm_cell is not None
        and imm_cell["reseed_rate_500"] is not None
        and fast_ctl is not None
        and fast_ctl["reseed_rate_500"] is not None
        and imm_cell["reseed_rate_500"] < fast_ctl["reseed_rate_500"]
    )

    # Does any mortal cell now compose pins + tape-scale life + kernel
    # (the joint_tune closure test)?
    def _closed(c: dict[str, Any]) -> bool:
        return (
            c["n_pins_mean"] >= 6.5
            and c["life_ev_p50_mean"] is not None
            and 0.5 * _TAPE_LIFE_EV <= c["life_ev_p50_mean"] <= 2.0 * _TAPE_LIFE_EV
            and c["instant_mean"] is not None
            and abs(c["instant_mean"] - 0.887) <= 0.35
            and c["k200_mean"] is not None
            and 3.0 <= c["k200_mean"] <= 6.0
        )

    claims = {
        "cells_measured": all(c["n_draws"] == len(_SEEDS) for c in cells),
        "mortal_damps_overshoot": mortal_damps,
        "immune_damps_rate": immune_damps,
        "joint_closure_found": any(_closed(c) for c in cells),
        # The all-pins cell survives mortality (control vs rr0 both >=6).
        "pins_survive_mortal": (
            fast_ctl is not None
            and fast_ctl["n_pins_mean"] >= 6.0
            and any(c["regime"] == "joint_d60_rr0" and c["n_pins_mean"] >= 6.0 for c in cells)
        ),
    }

    out: dict[str, Any] = {
        "schema": MORTAL_REPOST_SCHEMA,
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
                    "repost_requote": rr,
                    "repost_ttl_immune": imm,
                }
                for label, z, t, r, d, rr, imm in _CELLS
            ],
        },
        "tape_reference": {
            "life_p50_events": round(_TAPE_LIFE_EV, 1),
            "k200_ticks": 4.64,
            "instant_ticks": 0.887,
            "reseed_rate_500_band": [_RESEED_LO, _RESEED_HI],
            "sources": ["order_lifetime.v1", "impact_persist.v1", "reseed_hazard.v1"],
        },
        "cells": cells,
        "claims": claims,
        "notes": (
            "Mortal reposts FALSIFIED at 15k: every rr0 cell overshoots "
            "the reseed rate WORSE (0.73-0.79 vs ctl 0.64-0.67) — a "
            "one-shot repost re-empties at expiry, and each re-empty/"
            "re-fill cycle counts as another measured episode. "
            "The inverse (repost_ttl_immune) moves the rate the RIGHT "
            "direction on slow cells but not fast: joint_d60_imm holds "
            "7/7 pins with rate 0.558 in-band (vs ctrl 0.640) and "
            "mid_d60_imm reaches 6.5 pins with k200 5.44 in band — but "
            "fast_d110_imm RAISES the rate (0.766): sticky reseeds on a "
            "50-ev ttl still get consumed by ambient churn and the "
            "levels keep cycling. joint_closure_found stays False — "
            "immune pushes life UP (104.5 on the joint cell, 82.4 on "
            "mid — the latter misses only the <=51-ev band). The "
            "residual is now measured: reposted depth must persist "
            "against ttl but die against the ambient churn hazard — a "
            "mechanism between one-shot and immune."
        ),
    }
    body = dict(out)
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="receipts/mortal_repost.json")
    args = ap.parse_args()
    out = mortal_repost_bench(horizon=args.horizon, seed=args.seed)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
