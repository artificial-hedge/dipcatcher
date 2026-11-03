#!/usr/bin/env python3
"""List drifted corpus epochs and the owner commands that restamp them.

Read-only. Stdout is one JSON object. This process does not stamp, sign,
anchor, or checkpoint, and it does not write the heads pin or any receipt.
The corpora are exactly deploy, docs, docker, scripts, src, and tests.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

from quant_fund.research.corpus_epoch import load_heads_pin, member_digests, member_tree_root

CORPORA = ("deploy", "docs", "docker", "scripts", "src", "tests")
PATTERN = "*"
OWNER_STEPS = (
    "make stamp-epochs",
    "make sign-pins",
    "make anchor-pins",
    "make checkpoint",
)


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


def drift_report(
    pins: Mapping[str, Mapping[str, str]],
    trees: Mapping[str, str],
) -> dict[str, object]:
    """Which of the six corpora differ from the pinned tree, plus owner commands."""
    epochs: list[dict[str, object]] = []
    drifted: list[str] = []
    for name in CORPORA:
        pin = pins.get(f"{name}/{PATTERN}", {})
        pinned = str(pin.get("tree_root", ""))
        current = str(trees.get(name, ""))
        moved = not pinned or not current or pinned != current
        if moved:
            drifted.append(name)
        epochs.append(
            {
                "corpus": name,
                "drifted": moved,
                "pinned_tree_root": pinned,
                "receipt": str(pin.get("receipt", "")),
                "tree_root": current,
            }
        )
    return {
        "corpora": list(CORPORA),
        "drifted": drifted,
        "epochs": epochs,
        "owner_steps": list(OWNER_STEPS),
        "read_only": True,
        "signed": False,
    }


def live_trees(root: Path) -> dict[str, str]:
    """Current member-tree roots. Reads files. Does not stamp."""
    trees: dict[str, str] = {}
    for name in CORPORA:
        members = member_digests(root / name, pattern=PATTERN)
        trees[name] = member_tree_root(members)
    return trees


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        _emit({"error": "owner-restamp takes no arguments", "read_only": True, "signed": False})
        return 1
    root = Path(__file__).resolve().parents[1]
    try:
        pins = load_heads_pin(root / "quality" / "epoch_heads.json")
        report = drift_report(pins, live_trees(root))
    except (OSError, ValueError):
        _emit({"error": "epoch pin", "read_only": True, "signed": False})
        return 1
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
