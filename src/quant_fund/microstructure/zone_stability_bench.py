"""zone_stability — is the zone closure structural or a rate?

``zone_embargo.v1`` landed the campaign's first stable-looking
closure: z12 under iid held all seven pins on both draws. Two seeds
is a suggestive rate, not a claim — this bench runs the seed panel
the campaign's stability convention demands (``pin_stability``/
``floor_stability``: per-pin hold RATES over a seed panel, both
flows), plus the impact-kernel check on the closed cell — does the
no-quote zone carry the tape's drift profile too?

Receipts are sealed via ``receipt_sha256`` and carry ``data_label``
``"MIXED"`` — synthetic draws measured against committed real-tape
pin targets and kernel markers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_fund.microstructure.crown_density_bench import _sim_crown
from quant_fund.microstructure.full_impact_bench import _FULL
from quant_fund.microstructure.full_stack_bench import _PIN_FIELDS, _pins_ok
from quant_fund.microstructure.impact_persist_bench import _measure
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.reseed_hazard_bench import sim_reseed
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

ZONE_STABILITY_SCHEMA = "zone_stability.v1"

# The closure corner and its split-flow sibling: enough of a panel to
# tell structural from knife-edge.
_CONFIGS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "z12_d280",
        dict(
            _FULL,
            zone_embargo=12,
            fill_repost_frac=0.8,
            fill_repost_delay=280,
            repost_frac=0.6,
            repost_band=4,
            repost_window=500,
        ),
    ),
    (
        "z10_d280",
        dict(
            _FULL,
            zone_embargo=10,
            fill_repost_frac=0.8,
            fill_repost_delay=280,
            repost_frac=0.6,
            repost_band=4,
            repost_window=500,
        ),
    ),
)

_FLOWS: tuple[tuple[str, float | None], ...] = (("iid", None), ("split", 2.0))
_SEEDS = (7, 11, 101, 211)

# Tape kernel markers (impact_persist.v1 / touch_follow.v1): instant
# signed drift 0.887 ticks; continuation at +200 events 4.64.
_TAPE_INSTANT = 0.887
_TAPE_K200 = 4.64


def _cell(extra: dict[str, Any], *, horizon: int, seed: int, inten: float | None) -> dict[str, Any]:
    crown = _sim_crown(
        "z", extra, horizon=horizon, seed=seed, collect_counts=True, flow_intensity=inten
    )
    st = sim_reseed("z", extra, horizon=horizon, seed=seed, flow_intensity=inten)
    cell = {
        "spread_mean": crown["spread_mean"],
        "crown_share_of_visible": crown["crown_share_of_visible"],
        "empty_share": (
            round(crown["n_reveals"] / crown["n_fills"], 4) if crown["n_fills"] else None
        ),
        "hidden_fill_share": crown.get("hidden_fill_share"),
        "reveal_gap_ticks_mean": crown["reveal_gap_ticks_mean"],
        "reseed_rate_500": st["reseed_rate_500"],
        "reseed_as_touch_share": st["reseed_as_touch_share"],
    }
    cell["pins"] = _pins_ok(cell)
    cell["n_fills"] = crown["n_fills"]
    return cell


def _impact(
    extra: dict[str, Any], *, horizon: int, seed: int, inten: float | None
) -> dict[str, Any]:
    cfg = _calibrated(seed, extra)
    flow = _split(inten, seed) if inten is not None else None
    return _measure(cfg, flow, horizon)


def zone_stability_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    for cfg_label, extra in _CONFIGS:
        for flow_label, inten in _FLOWS:
            draws: list[dict[str, Any]] = []
            for s in _SEEDS:
                cell = _cell(extra, horizon=horizon, seed=seed * 1000 + s, inten=inten)
                draws.append(
                    {
                        "run_seed": seed * 1000 + s,
                        "n_pins_ok": sum(cell["pins"].values()),
                        "all_seven": sum(cell["pins"].values()) == len(cell["pins"]),
                        "pins": cell["pins"],
                        "spread_mean": cell["spread_mean"],
                        "empty_share": cell["empty_share"],
                        "n_fills": cell["n_fills"],
                    }
                )
            per_pin: dict[str, float] = {}
            for pin, _f in _PIN_FIELDS:
                per_pin[pin] = round(
                    sum(1 for d in draws if d["pins"].get(pin, False)) / len(draws), 4
                )
            panels.append(
                {
                    "config": cfg_label,
                    "flow": flow_label,
                    "n_draws": len(draws),
                    "per_pin_rate": per_pin,
                    "all_seven_rate": round(
                        sum(1 for d in draws if d["all_seven"]) / len(draws), 4
                    ),
                    "mean_pins_ok": round(sum(d["n_pins_ok"] for d in draws) / len(draws), 3),
                    "draws": draws,
                }
            )

    # Impact kernel on the closed cell (iid and split) — the channel
    # that pinned all prior compositions.
    kernels: list[dict[str, Any]] = []
    for flow_label, inten in _FLOWS:
        k = _impact(_CONFIGS[0][1], horizon=horizon, seed=seed * 1000 + 7, inten=inten)
        kernels.append({"config": "z12_d280", "flow": flow_label, **k})

    claims = {
        "cells_measured": all(p["n_draws"] == len(_SEEDS) for p in panels),
        # Structural claim: some (config, flow) panel holds all seven
        # pins at rate 1.0 across the four-seed panel.
        "zone_closure_structural": any(p["all_seven_rate"] >= 1.0 for p in panels),
        # Weaker: closure at rate >= 0.5 somewhere.
        "zone_closure_majority": any(p["all_seven_rate"] >= 0.5 for p in panels),
        # The iid closure cell also carries the instant pin's channel:
        # |instant - 0.887| <= 0.35 on at least one kernel draw.
        "zone_carries_instant": any(
            abs(k["instant_signed_ticks"] - _TAPE_INSTANT) <= 0.35
            for k in kernels
            if k.get("instant_signed_ticks") is not None
        ),
    }

    out: dict[str, Any] = {
        "schema": ZONE_STABILITY_SCHEMA,
        "kind": "microstructure_bench",
        "research_only": True,
        "data_label": "MIXED",
        "git_revision": git_revision(),
        "horizon": horizon,
        "seed": seed,
        "config": {
            "seeds": list(_SEEDS),
            "flows": [f for f, _ in _FLOWS],
            "configs": [c for c, _ in _CONFIGS],
        },
        "panels": panels,
        "impact_kernels": kernels,
        "tape_kernel": {"instant_ticks": _TAPE_INSTANT, "k200_ticks": _TAPE_K200},
        "claims": claims,
        "notes": (
            "Seed-panel stability of the zone_embargo closure "
            "(zone_embargo.v1's z12-iid 7/7@2-seeds): four seeds x two "
            "flows x two zone heights, per-pin hold rates plus "
            "all-seven rate. Impact kernels on the z12 cell check "
            "whether the closed composition also carries the tape's "
            "instant (0.887) / continuation (4.64) drift markers. "
            "Prior closures were knife-edges (joint_stability 0/4, "
            "floor cells ~0.5-0.75 spread rate). Measured at 15k: "
            "z12-iid holds all-seven at rate 0.75 (3/4 draws) — six "
            "pins structural at rate 1.0, only reveal_gap fragile "
            "(0.75). zone_closure_majority TRUE; the strict 1.0 "
            "structural claim stays FALSE (honest rate). The same "
            "cell carries the drift kernel under iid: instant 1.048 "
            "vs tape 0.887, k200 5.20 vs 4.64. Under split the empty "
            "pin collapses (rate 0.0) — metaorders over-empty the "
            "wide zone."
        ),
    }
    body = dict(out)
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="receipts/zone_stability.json")
    args = ap.parse_args()
    out = zone_stability_bench(horizon=args.horizon, seed=args.seed)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
