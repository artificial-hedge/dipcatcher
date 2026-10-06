"""grammar_map — sealed synthesis of the wave-25 event-grammar campaign.

``zone_map.v1`` closed the standing-spread geometry (7/7 pins on the
zone cell) — but ``zone_card`` measured the closed cell against the
tape's event grammar and found the divergence had moved: maker
lifetimes sat ~6-12x the tape's ~25.5-event median. This map
re-verifies the grammar-campaign receipts on top of the closed zone
cell and records the final verdict.

This is a synthesis bench, not a measurement: it re-verifies each
member receipt's seal before reading its claims, so the map is only as
honest as the receipts underneath — and the seal check is in the
claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

GRAMMAR_MAP_SCHEMA = "grammar_map.v1"

# The wave-25 lanes and the question each one answered.
_LANES: tuple[tuple[str, str], ...] = (
    ("zone_card.json", "is the closed cell's maker grammar tape-scale?"),
    ("zone_ttl.json", "does maker aging (ttl) bracket the residual?"),
    ("zone_churn.json", "does cancel-into-repost churn close it?"),
    ("churn_stability.json", "is the closure structural or a rate?"),
    ("churn_reseed.json", "where do emptied vacancies die?"),
    ("repost_latency.json", "does repost delay resolve reseed residual?"),
    ("joint_tune.json", "is the pins/life corner a frontier?"),
)


def grammar_map(receipts_dir: Path | str = "receipts") -> dict[str, Any]:
    """Compose the grammar-campaign receipts into the final map."""
    # Lazy: the verifier lives in the research layer above microstructure;
    # importing under a private alias is the sanctioned way to break that edge.
    from quant_fund.research.receipt_v2 import (  # noqa: PLC0415
        verify_receipt_file as _verify_receipt_file,
    )

    receipts_dir = Path(receipts_dir)
    lanes: list[dict[str, Any]] = []
    all_sealed = True
    for fname, question in _LANES:
        path = receipts_dir / fname
        entry: dict[str, Any] = {"receipt": fname, "question": question}
        if not path.exists():
            entry["sealed"] = False
            entry["claims"] = None
            all_sealed = False
        else:
            entry["sealed"] = bool(_verify_receipt_file(path)["valid"])
            all_sealed = all_sealed and entry["sealed"]
            entry["claims"] = json.loads(path.read_text()).get("claims", {})
        lanes.append(entry)

    found = {e["receipt"]: e for e in lanes if e["claims"] is not None}
    churn = found.get("zone_churn.json")
    stab = found.get("churn_stability.json")
    reseed = found.get("churn_reseed.json")
    latency = found.get("repost_latency.json")
    tune = found.get("joint_tune.json")

    claims = {
        "all_member_receipts_sealed": bool(all_sealed),
        "all_lanes_present": len(lanes) == len(_LANES)
        and all(e["claims"] is not None for e in lanes),
        # The grammar gap was real: the closed-geometry cell did not
        # carry tape-scale maker lifetimes.
        "grammar_gap_was_real": bool(
            found.get("zone_card.json", {}).get("claims", {}).get("life_on_tape_scale") is False
        ),
        # Aging alone reaches the tape scale without breaking closure.
        "ttl_reaches_scale": bool(
            found.get("zone_ttl.json", {}).get("claims", {}).get("ttl_reaches_life_scale")
        ),
        # Churn (ttl + requote) holds all 7 pins with maker scale.
        "churn_closes": bool(churn is not None and churn["claims"].get("churn_closes_joint")),
        # But churn is NOT the pin mechanism (rq0 controls hold too).
        "churn_not_mechanism": bool(
            stab is not None and stab["claims"].get("churn_is_mechanism") is False
        ),
        # The residual was diagnosed: repost under-engagement, not
        # walked-past drops.
        "residual_is_underfill": bool(
            reseed is not None and reseed["claims"].get("repost_underfill")
        ),
        # The delay sweep landed a new all-pins cell carrying the kernel.
        "latency_closes": bool(
            latency is not None
            and latency["claims"].get("pins_survive")
            and latency["claims"].get("kernel_carried")
        ),
        # Final verdict: pins + tape-scale life + kernel do NOT compose
        # on one cell — the corner is a Pareto frontier (~half-pin gap).
        "joint_frontier_exists": bool(
            tune is not None
            and tune["claims"].get("joint_closure_found") is False
            and tune["claims"].get("grammar_keeps_pins")
        ),
    }

    falsified = [
        "pure maker aging as pin-lifter (zone_ttl — ttl brackets but does not lift)",
        "churn as pin mechanism (churn_stability — rq0 controls hold equally)",
        "walked-past repost drops as binder (churn_reseed — rare and benign)",
        "pins + grammar + kernel on one cell (joint_tune — Pareto frontier)",
    ]
    payload: dict[str, Any] = {
        "schema": GRAMMAR_MAP_SCHEMA,
        "kind": "wave25_map",
        "asset": "AMZN",
        "n_lanes": len(lanes),
        "lanes": lanes,
        "mechanism_map": {
            "maker_aging": "maker_ttl — cancel path expiry, 0 = bit-identical",
            "requote_churn": "maker_requote — cancel-into-repost on expiry",
            "reseed_timing": "fill_repost_delay — d60 holds pins, d110 hits tape grammar",
            "frontier": "ttl<=75 for tape life vs ttl>=150 for 7 pins",
        },
        "falsified_channels": falsified,
        "claims": claims,
        "interpretation": (
            "Event-grammar campaign closed as a mapped frontier. "
            "zone_card found the divergence: the closed-spread cell's "
            "executed makers lived ~150-300 events vs the tape's ~25.5. "
            "maker_ttl brackets it (tape-speed aging reaches 35.8 ev and "
            "keeps 6/7 pins); maker_requote resolves it (z12+ttl200+rq60 "
            "holds all 7 pins at life ~88 ev, and ttl50+rq90 sits at the "
            "tape's exact median, 26.6 vs 25.5). churn_reseed instrumented "
            "the last channel: emptied-vacancy deaths are benign refills, "
            "and the binder was repost under-engagement — the d60 delay "
            "lands a new all-pins cell. joint_tune maps the final corner: "
            "pins + tape-scale life + kernel form a Pareto frontier — "
            "fast churn overshoots the reseed pin (vacancies kept hot) "
            "while slow churn holds it, and the closest compose "
            "(ttl50_rq90_d110) misses by ~half a pin. Every channel now "
            "has its mechanism AND its measured failure mode; the "
            "remaining divergence is a rate phenomenon, not a knob gap."
        ),
        "git_revision": git_revision(),
        "data_label": "MIXED",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["GRAMMAR_MAP_SCHEMA", "grammar_map"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts-dir", default="receipts")
    ap.add_argument("--out", default="receipts/grammar_map.json")
    args = ap.parse_args()
    out = grammar_map(receipts_dir=args.receipts_dir)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
