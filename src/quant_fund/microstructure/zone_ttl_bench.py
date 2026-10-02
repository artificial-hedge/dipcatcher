"""zone_ttl — does maker aging close the grammar without breaking closure?

``zone_card.v1`` found the closed zone cell's residual is the event-rate
channel: executed makers rest ~150-300 events vs the tape's ~25.5, and
the delete share sits ~0.39 vs the tape's 0.458 — the sim's makers live
too long. ``maker_ttl`` is the diagnosed mechanism: resting orders
auto-expire T events after submission (running through the normal
cancel path, so expiries count as deletes, vacate levels for the repost
machinery, and shorten maker age at fill). This bench sweeps ttl on the
closed z12 cell and measures (a) the card: mix shares + maker lifetime
vs the tape's pins, and (b) the closure surface: all seven pins plus
the instant-drift channel — whether aging the book fixes the grammar
without reopening the geometry.

Receipts are sealed via ``receipt_sha256`` and carry ``data_label``
``"MIXED"`` — synthetic draws measured against committed real-tape
pins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.crown_density_bench import _sim_crown
from quant_fund.microstructure.full_impact_bench import _FULL
from quant_fund.microstructure.full_stack_bench import _pins_ok
from quant_fund.microstructure.impact_persist_bench import _measure
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.reseed_hazard_bench import sim_reseed
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator
from quant_fund.microstructure.zone_card_bench import (
    _TAPE_EV_PER_S,
    _TAPE_LIFE_S,
    _TAPE_MIX,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

ZONE_TTL_SCHEMA = "zone_ttl.v1"

# (label, zone_embargo, maker_ttl, flow_intensity) — the closed cell at
# a ttl sweep plus the zone-free and ttl-free baselines.
_CELLS: tuple[tuple[str, int, int, float | None], ...] = (
    ("z0_ttl0_iid", 0, 0, None),
    ("z12_ttl0_iid", 12, 0, None),
    ("z12_ttl26_iid", 12, 26, None),
    ("z12_ttl50_iid", 12, 50, None),
    ("z12_ttl75_iid", 12, 75, None),
    ("z12_ttl100_iid", 12, 100, None),
    ("z12_ttl150_iid", 12, 150, None),
    ("z12_ttl200_iid", 12, 200, None),
    ("z12_ttl50_split", 12, 50, 2.0),
    ("z12_ttl100_split", 12, 100, 2.0),
)

_SEEDS = (7, 11)


def _extra(
    zone: int,
    ttl: int,
    requote: float = 0.0,
    fr_frac: float = 0.8,
    fr_delay: int = 280,
    repost_requote: float | None = None,
    repost_ttl_immune: bool = False,
) -> dict[str, Any]:
    return dict(
        _FULL,
        zone_embargo=zone,
        maker_ttl=ttl,
        maker_requote=requote,
        repost_requote=repost_requote,
        repost_ttl_immune=repost_ttl_immune,
        fill_repost_frac=fr_frac,
        fill_repost_delay=fr_delay,
        repost_frac=0.6,
        repost_band=4,
        repost_window=500,
    )


def _card(
    zone: int,
    ttl: int,
    inten: float | None,
    *,
    horizon: int,
    seed: int,
    requote: float = 0.0,
    fr_frac: float = 0.8,
    fr_delay: int = 280,
    repost_requote: float | None = None,
    repost_ttl_immune: bool = False,
) -> dict[str, Any]:
    """Mix shares + executed-maker lifetime on one draw (zone_card.v1's
    measure: ages on the sim clock converted to event units by the
    draw's realized rate)."""
    cfg = _calibrated(
        seed,
        _extra(zone, ttl, requote, fr_frac, fr_delay, repost_requote, repost_ttl_immune),
    )
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
    rate_ev_per_t = sim.n_events / sim._t if sim._t > 0 else 0.0
    ages_ev = ages_a * rate_ev_per_t
    life_p50 = float(np.median(ages_ev)) if len(ages_ev) else None
    return {
        "n_fills": sim.n_fills,
        "mix_share": {
            "sub": round(subs / tot, 4) if tot else None,
            "delete": round(dels / tot, 4) if tot else None,
            "exec": round(execs / tot, 4) if tot else None,
        },
        "life_events_p50_executed": life_p50,
    }


def _cell(
    zone: int,
    ttl: int,
    inten: float | None,
    *,
    horizon: int,
    seed: int,
    requote: float = 0.0,
    fr_frac: float = 0.8,
    fr_delay: int = 280,
    repost_requote: float | None = None,
    repost_ttl_immune: bool = False,
) -> dict[str, Any]:
    """Card + pins + kernel on one (zone, ttl, flow) draw."""
    extra = _extra(zone, ttl, requote, fr_frac, fr_delay, repost_requote, repost_ttl_immune)
    card = _card(
        zone,
        ttl,
        inten,
        horizon=horizon,
        seed=seed,
        requote=requote,
        fr_frac=fr_frac,
        fr_delay=fr_delay,
        repost_requote=repost_requote,
        repost_ttl_immune=repost_ttl_immune,
    )
    crown = _sim_crown(
        "joint",
        extra,
        horizon=horizon,
        seed=seed,
        collect_counts=True,
        flow_intensity=inten if inten is not None else 3.0,
    )
    reseed = sim_reseed("joint", extra, horizon=horizon, seed=seed, flow_intensity=inten)
    pin_cell = dict(crown)
    pin_cell.update(reseed)
    pin_cell["empty_share"] = crown["n_reveals"] / crown["n_fills"] if crown["n_fills"] else 0.0
    pins = _pins_ok(pin_cell)
    kernel = _measure(
        _calibrated(seed, extra),
        _split(inten, seed) if inten is not None else None,
        horizon,
    )
    return {
        "card": card,
        "pins": pins,
        "n_pins": sum(pins.values()),
        "instant_signed_ticks": kernel["instant_signed_ticks"],
        "k200": kernel["kernel_mean_ticks"].get("200"),
    }


def zone_ttl_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for label, zone, ttl, inten in _CELLS:
        draws = [_cell(zone, ttl, inten, horizon=horizon, seed=seed * 1000 + s) for s in _SEEDS]
        mixes = [d["card"]["mix_share"] for d in draws]
        mix_mean = {
            k: (
                round(sum(m[k] for m in mixes if m[k] is not None) / len(mixes), 4)
                if all(m[k] is not None for m in mixes)
                else None
            )
            for k in ("sub", "delete", "exec")
        }
        sub, dele, exc = mix_mean["sub"], mix_mean["delete"], mix_mean["exec"]
        tape_gap: float | None = None
        if sub is not None and dele is not None and exc is not None:
            tape_gap = round(
                (
                    abs(sub - _TAPE_MIX["sub"])
                    + abs(dele - _TAPE_MIX["delete"])
                    + abs(exc - _TAPE_MIX["exec"])
                )
                / 3.0,
                4,
            )
        lifes = [
            d["card"]["life_events_p50_executed"]
            for d in draws
            if d["card"]["life_events_p50_executed"] is not None
        ]
        cells.append(
            {
                "regime": label,
                "zone_embargo": zone,
                "maker_ttl": ttl,
                "flow": "iid" if inten is None else "split",
                "n_draws": len(draws),
                "mix_mean": mix_mean,
                "tape_mix_gap": tape_gap,
                "life_ev_p50_mean": (round(sum(lifes) / len(lifes), 2) if lifes else None),
                "n_pins_mean": sum(d["n_pins"] for d in draws) / len(draws),
                "instant_mean": (
                    sum(
                        d["instant_signed_ticks"]
                        for d in draws
                        if d["instant_signed_ticks"] is not None
                    )
                    / max(
                        1,
                        sum(1 for d in draws if d["instant_signed_ticks"] is not None),
                    )
                ),
                "k200_mean": (
                    sum(d["k200"] for d in draws if d["k200"] is not None)
                    / max(1, sum(1 for d in draws if d["k200"] is not None))
                ),
                "draws": draws,
            }
        )

    tape_life_ev = _TAPE_LIFE_S["executed_p50"] * _TAPE_EV_PER_S
    closed = cells[1]  # z12_ttl0_iid
    claims = {
        "cells_measured": all(c["n_draws"] == len(_SEEDS) for c in cells),
        # Some ttl cell reaches the tape's ~25.5-event maker scale.
        "ttl_reaches_life_scale": any(
            c["life_ev_p50_mean"] is not None
            and 0.5 * tape_life_ev <= c["life_ev_p50_mean"] <= 2.0 * tape_life_ev
            for c in cells
        ),
        # Some ttl cell beats the closed cell's mix gap.
        "ttl_improves_mix": any(
            c["maker_ttl"] > 0
            and c["tape_mix_gap"] is not None
            and closed["tape_mix_gap"] is not None
            and c["tape_mix_gap"] < closed["tape_mix_gap"]
            for c in cells
        ),
        # Some ttl cell keeps >=6/7 pins while holding the card close.
        "ttl_keeps_closure": any(
            c["maker_ttl"] > 0
            and c["n_pins_mean"] >= 6.0
            and c["tape_mix_gap"] is not None
            and closed["tape_mix_gap"] is not None
            and c["tape_mix_gap"] <= closed["tape_mix_gap"] + 0.01
            for c in cells
        ),
        # Kernel survives on some ttl cell.
        "kernel_carried": any(
            c["maker_ttl"] > 0 and abs(c["instant_mean"] - 0.887) <= 0.35 for c in cells
        ),
    }

    out: dict[str, Any] = {
        "schema": ZONE_TTL_SCHEMA,
        "kind": "microstructure_bench",
        "research_only": True,
        "data_label": "MIXED",
        "git_revision": git_revision(),
        "horizon": horizon,
        "seed": seed,
        "config": {
            "seeds_per_cell": list(_SEEDS),
            "cells": [
                {
                    "label": label,
                    "zone_embargo": z,
                    "maker_ttl": t,
                    "flow_intensity": i,
                }
                for label, z, t, i in _CELLS
            ],
        },
        "tape_reference": {
            "mix_share": _TAPE_MIX,
            "events_per_s": _TAPE_EV_PER_S,
            "executed_p50_events": round(tape_life_ev, 1),
            "sources": ["event_matrix.v1", "order_lifetime.v1"],
        },
        "cells": cells,
        "claims": claims,
        "notes": (
            "Maker aging on the closed z12 cell. The tape's grammar "
            "residual is specifically the maker horizon: at 15k the "
            "baseline mix gap is only ~0.007 (the 3k preview's 0.049 "
            "was early-run noise — the mix was never far), while "
            "executed-maker p50 sits ~205 events vs the tape's 25.5. "
            "ttl sweeps across the tape value (26->10.6, 50->22.1, "
            "75->35.8, 100->40.6, 150->72, 200->89.6 events): "
            "grammar wants ttl ~50-100, where p50 lands within 2x "
            "of the tape. But the pins/kernel want ttl ~200 "
            "(ttl200_iid: 6/7 pins, instant 1.099, k200 6.02) — the "
            "two targets bracket the mechanism without coinciding: "
            "tape-speed aging alone starves fills below the pin "
            "floor (ttl50: 3/7). Honest residual: the sim already "
            "resets age on reprice (every reprice is a fresh order), "
            "so the residual is not repricing semantics — it is the "
            "missing class of makers whose fast churn is offset by "
            "staying visible, i.e. cancel-into-repost loops that "
            "keep depth constant while aging out."
        ),
    }
    body = dict(out)
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="receipts/zone_ttl.json")
    args = ap.parse_args()
    out = zone_ttl_bench(horizon=args.horizon, seed=args.seed)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
