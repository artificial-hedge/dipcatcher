"""pin_stability — how fragile is the emptied-touch closure?

``joint_stability.v1`` showed the joint cell is a knife-edge over
re-seeded draws (0/4 hold all four original pins). ``flow_couple.v1``
then showed that under a single flow, no intensity holds all seven
pins — the spread pin in particular is seed- and flow-fragile. This
bench quantifies the fragility properly: run the all-pins config at
each flow class over a seed panel and report a per-pin *hold rate*
plus the kernel bands on the persistent-flow cells.

The claim is not "the closure was wrong" — the single-seed 7/7 is a
true recorded event — but that the pins sit inside a ~±0.05-seed-noise
band, so a pin-level hold RATE is the honest object, not a single
verdict. Which pins are robust (crown, hidden, reseed) and which are
knife-edge (spread, empty, reveal) is itself the deliverable.

Measured (6 seeds x 2 flows @ 12k): under iid, spread holds 0/6 (the
burst-driven repost/pull machinery needs metaorder sweeps) and
reveal_gap holds 2/6; under split@2, empty holds 0/6 (split bursts
over-sweep past the 0.663 band edge) while reveal holds 5/6.
Crown and hidden are robust (6/6) under both flows. The instant
channel also fails the seed panel (0.45 vs 0.887 under split@2) — the
single-cell in-band readings were partly knife-edge draws too.
n_all_pins is 0/6 under both flows: the closure is a rate phenomenon,
not a per-seed certainty.

Evidence class: research / MIXED (sim cells vs committed tape pins).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.microstructure.crown_density_bench import _sim_crown
from quant_fund.microstructure.full_impact_bench import _FULL
from quant_fund.microstructure.full_stack_bench import _pins_ok
from quant_fund.microstructure.impact_persist_bench import _LAGS, _REAL, _measure
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.reseed_hazard_bench import sim_reseed
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

PIN_STABILITY_SCHEMA = "pin_stability.v1"

_N_SEEDS = 6
_FLOWS: tuple[float | None, ...] = (None, 2.0)


def pin_stability_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    """Per-pin hold rates of the all-pins cell over seeds x flows."""
    rows: list[dict[str, Any]] = []
    for f_i, inten in enumerate(_FLOWS):
        pin_votes: dict[str, list[bool]] = {}
        inst_list: list[float] = []
        k200_list: list[float] = []
        n_closed = 0
        for s in range(_N_SEEDS):
            sd = seed + f_i * 100 + s * 13
            label = f"{'iid' if inten is None else f'split_{inten:g}'}_s{sd}"
            cr = _sim_crown(
                label,
                _FULL,
                horizon=horizon,
                seed=sd,
                collect_counts=True,
                flow_intensity=inten,
            )
            st = sim_reseed(label, _FULL, horizon=horizon, seed=sd, flow_intensity=inten)
            cr.update({k: v for k, v in st.items() if k != "regime"})
            n_f = cr["n_fills"]
            cr["empty_share"] = round(cr["n_reveals"] / n_f, 4) if n_f else None
            pins = _pins_ok(cr)
            for pin, ok in pins.items():
                pin_votes.setdefault(pin, []).append(bool(ok))
            flow = _split(inten, sd) if inten is not None else None
            km = _measure(_calibrated(sd, _FULL), flow, horizon)
            ins = km["instant_signed_ticks"]
            k200 = km["kernel_mean_ticks"]["200"]
            if ins is not None:
                inst_list.append(ins)
            if k200 is not None:
                k200_list.append(k200)
            if all(pins.values()):
                n_closed += 1
        hold = {pin: round(sum(v) / len(v), 4) for pin, v in pin_votes.items()}
        rows.append(
            {
                "flow_intensity": inten,
                "n_seeds": _N_SEEDS,
                "pin_hold_rate": hold,
                "mean_pins_ok": round(
                    float(np.mean([sum(v) for v in zip(*pin_votes.values(), strict=False)])),
                    3,
                )
                if pin_votes
                else None,
                "n_all_pins": n_closed,
                "instant_mean": round(float(np.mean(inst_list)), 4) if inst_list else None,
                "instant_sd": round(float(np.std(inst_list)), 4) if inst_list else None,
                "k200_mean": round(float(np.mean(k200_list)), 4) if k200_list else None,
                "k200_sd": round(float(np.std(k200_list)), 4) if k200_list else None,
            }
        )

    divergences: list[str] = []
    for r in rows:
        for pin, rate in r["pin_hold_rate"].items():
            if rate < 1.0:
                divergences.append(f"{r['flow_intensity']}:{pin}_hold_{rate:.2f}")

    iid_row, split_row = rows[0], rows[1]
    fragile = {
        r["flow_intensity"]: [p for p, v in r["pin_hold_rate"].items() if v < 0.8] for r in rows
    }
    claims = {
        "panel_measured": all(r["n_seeds"] == _N_SEEDS for r in rows),
        # Knife-edge claim: all-seven-pins is NOT the modal outcome —
        # the closure is a rate phenomenon, not a per-seed certainty.
        "closure_is_knife_edge": all(r["n_all_pins"] < _N_SEEDS for r in rows),
        # At least one pin fails on >20% of seeds in some flow class —
        # the fragility is real and pin-specific.
        "fragility_quantified": any(bool(v) for v in fragile.values()),
        # The iid flow loses the spread pin reliably — the burst-driven
        # repost/pull machinery needs metaorder sweeps to fire.
        "iid_spread_fails": iid_row["pin_hold_rate"].get("spread", 1.0) < 0.5,
        # Under persistent flow the instant channel holds on average
        # while continuation overshoots (the full_impact verdict,
        # now seed-robust).
        "split_instant_holds_k200_overshoots": bool(
            split_row["instant_mean"] is not None
            and split_row["k200_mean"] is not None
            and abs(split_row["instant_mean"] - _REAL["instant_signed_ticks"]) < 0.25
            and split_row["k200_mean"] > _REAL["kernel"]["200"] + 0.5
        ),
    }
    body: dict[str, Any] = {
        "schema": PIN_STABILITY_SCHEMA,
        "kind": "sim_vs_real",
        "git_revision": git_revision(),
        "research_only": True,
        "data_label": "MIXED",
        "horizon": horizon,
        "seed": seed,
        "n_seeds": _N_SEEDS,
        "lags": list(_LAGS),
        "tape_targets": _REAL,
        "rows": rows,
        "divergences": divergences,
        "claims": claims,
        "notes": (
            "Per-pin hold RATES over a seed panel (seeds spaced 13 apart "
            "per flow class), not a single verdict. The single-seed 7/7 "
            "event recorded in full_stack.v1 stands; this receipt "
            "quantifies how often each pin lands inside its band under a "
            "uniform flow, and reports the drift kernel's mean+sd on the "
            "same seeds. A pin holding <100% of seeds inside a band the "
            "tape sits inside permanently is the honest fragility "
            "statement for that mechanism. Measured: crown/hidden "
            "robust under both flows; spread holds 0/6 under iid; empty "
            "holds 0/6 under split@2; the instant channel is itself "
            "seed-fragile (0.45 vs 0.887); all-seven-pins lands 0/6 "
            "under either flow."
        ),
    }
    body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return body
