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
from quant_fund.microstructure.joint_tune_contract import (
    _CELLS,
    _SEEDS,
    claim_semantics,
    draw_closed,
    summaries,
    tape_reference,
)
from quant_fund.microstructure.joint_tune_contract import (
    _TAPE_INSTANT as _TAPE_INSTANT,
)
from quant_fund.microstructure.joint_tune_contract import (
    _TAPE_LIFE_EV as _TAPE_LIFE_EV,
)
from quant_fund.microstructure.joint_tune_contract import (
    claims as derive_claims,
)
from quant_fund.microstructure.zone_ttl_bench import _cell
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_draw_closed = draw_closed
JOINT_TUNE_SCHEMA = "joint_tune.v2"


def joint_tune_bench(*, horizon: int = 15000, seed: int = 7) -> dict[str, Any]:
    if type(horizon) is not int or horizon <= 0 or type(seed) is not int or seed < 0:
        raise ValueError(
            "joint_tune requires a positive integer horizon and nonnegative integer seed"
        )
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
                uniform_flow=True,
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
                record_inputs=True,
            )
            for s in _SEEDS
        ]

        for draw, fate in zip(draws, fates, strict=True):
            draw["reseed_fates"] = {
                k: fate[k]
                for k in (
                    "inputs",
                    "reseed_rate_500",
                    "reseed_as_touch_share",
                    "reseed_latency_p50",
                )
            }
        cells.append(
            {
                "regime": label,
                "zone_embargo": zone,
                "maker_ttl": ttl,
                "maker_requote": rq,
                "fill_repost_delay": fd,
                "draws": draws,
                **summaries(draws),
            }
        )
    claims = derive_claims(cells)

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
        "tape_reference": tape_reference(),
        "cells": cells,
        "claims": claims,
        "claim_semantics": claim_semantics(),
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
