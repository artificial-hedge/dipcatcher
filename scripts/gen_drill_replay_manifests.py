"""Generate committed ``replay_manifest.v1`` carriers for the real-tape drills.

Each committed ``receipts/*_real_drill*.json`` receipt is immutable evidence;
this generator emits a *separate* sealed carrier receipt declaring how the
drill re-executes — argv, the artifact bytes it must produce, and the input
tape manifest it binds to. ``dipcatcher replay <carrier>`` then re-runs the
drill against the pinned tape and re-hashes the artifacts.

The carrier also records claim-level equivalence with the committed receipt
it reproduces: ``reproduces.claims_equal`` compares every field except the
volatile set (``receipt_sha256``, ``code_revision``, ``meta``, frame-CSV
``inputs_sha256`` — environment-dependent encodings that never belonged in
sealed bytes; see the meta-strip convention). A committed receipt whose
claims legitimately drift — the fleet registry grew since it was sealed —
is recorded honestly with ``claims_equal: false`` and the differing keys.

Usage: ``PYTHONPATH=src python scripts/gen_drill_replay_manifests.py``
runs every drill under ``--out data/metadata/replay`` and seals carriers
into ``receipts/replay_manifest_<lane>.json``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

TAPE_MANIFEST = "data/manifests/yahoo_eod_us_wide.json"
REPLAY_OUT = "data/metadata/replay"
TAPE_ARG = "data/file_us_wide/bronze/bars.parquet"

# Fields excluded from claim-equality — volatile stamps and frame-hash
# encodings that differ across environments by design.
VOLATILE_KEYS = frozenset(
    {
        "receipt_sha256",
        "code_revision",
        "meta",
        "generated_at",
        "generated_at_commit",
        "inputs_sha256",
        "dataset_sha256",
        "created_at",
    }
)

# Committed receipts with no producer script — composite/manual artifacts.
UNREPRODUCIBLE: dict[str, str] = {
    "corpus_real_drill.json": (
        "rolling audit — its input is the live receipts corpus, which every "
        "new evidence file mutates; byte-stability is impossible by design. "
        "Corpus integrity is carried by the corpus epoch chains instead."
    ),
    "emerge_real_drill.json": (
        "composite emerge lane over per-lane streams — its inputs are the "
        "other drills' outputs, not a tape; recompute via the lane drill set"
    ),
    "honest_verdict_real_drill.json": (
        "superseded by verdict_real_drill.json — produced by a pre-convention "
        "writer that never entered scripts/"
    ),
}

# producer script -> committed receipt basenames it must regenerate
DRILLS: dict[str, dict[str, Any]] = {
    "calib_real_drill.py": {"receipts": ["calib_real_drill.json"], "argv_extra": []},
    "conformal_real_drill.py": {
        "receipts": [
            "conformal_real_drill_gaussian_pit.json",
            "conformal_real_drill_gaussian_minus_conf_t_pinball.json",
        ],
        "argv_extra": [],
    },
    "coverage_real_drill.py": {"receipts": ["coverage_real_drill.json"], "argv_extra": []},
    "coverage_cs_real_drill.py": {
        "receipts": ["coverage_cs_real_drill.json"],
        "argv_extra": [],
    },
    "cp_real_drill.py": {
        "receipts": [
            "cp_real_drill_gaussian_pit.json",
            "cp_real_drill_gaussian_minus_conf_t_pinball.json",
        ],
        "argv_extra": [],
    },
    "drift_real_drill.py": {
        "receipts": [
            "drift_real_drill_gaussian_pit.json",
            "drift_real_drill_gaussian_minus_conf_t_pinball.json",
        ],
        "argv_extra": [],
    },
    "loss_cs_real_drill.py": {
        "receipts": [
            "loss_cs_real_drill_gaussian_vs_conf_t.json",
            "loss_cs_real_drill_conf_t_vs_empirical.json",
        ],
        "argv_extra": [],
    },
    "mcs_real_drill.py": {"receipts": ["mcs_real_drill.json"], "argv_extra": []},
    "panel_audit_drill.py": {
        "receipts": ["panel_audit_real_drill.json"],
        "argv_extra": [],
    },
    "tail_real_drill.py": {"receipts": ["tail_real_drill.json"], "argv_extra": []},
    "verdict_real_drill.py": {"receipts": ["verdict_real_drill.json"], "argv_extra": []},
}


def _norm(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _norm(v) for k, v in node.items() if k not in VOLATILE_KEYS}
    if isinstance(node, list):
        return [_norm(v) for v in node]
    return node


def _claims_equal(committed: dict[str, Any], produced: dict[str, Any]) -> tuple[bool, list[str]]:
    left, right = _norm(committed), _norm(produced)
    for body in (left, right):
        drill = body.get("drill")
        if isinstance(drill, dict):
            drill.pop("tape", None)  # abs path on legacy receipts
    keys = sorted(set(left) | set(right))
    diff = [k for k in keys if left.get(k) != right.get(k)]
    return not diff, diff


def _seal_carrier(body: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(body)
    sealed["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return sealed


def main() -> int:
    root = Path.cwd()
    out_dir = root / REPLAY_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    us_wide = root / TAPE_MANIFEST
    if not us_wide.is_file():
        raise SystemExit(f"tape manifest missing: {us_wide}")

    carriers: list[str] = []
    failures: list[str] = []

    for script, spec in DRILLS.items():
        argv = ["scripts/" + script, TAPE_ARG, "--out", REPLAY_OUT, *spec["argv_extra"]]
        proc = subprocess.run(
            [sys.executable, *argv], cwd=root, capture_output=True, text=True, timeout=900
        )
        if proc.returncode != 0:
            failures.append(f"{script}: exit {proc.returncode}: {proc.stderr[-400:]}")
            continue
        artifacts = []
        reproduces = []
        for name in spec["receipts"]:
            produced = out_dir / name
            if not produced.is_file():
                failures.append(f"{script}: expected artifact {produced} not written")
                continue
            artifacts.append(
                {
                    "path": f"{REPLAY_OUT}/{name}",
                    "sha256": hashlib.sha256(produced.read_bytes()).hexdigest(),
                }
            )
            committed = root / "receipts" / name
            if committed.is_file():
                equal, diff = _claims_equal(
                    json.loads(committed.read_text()), json.loads(produced.read_text())
                )
                reproduces.append(
                    {
                        "receipt": f"receipts/{name}",
                        "committed_sha256": hashlib.sha256(committed.read_bytes()).hexdigest(),
                        "claims_equal": equal,
                        "claim_diff_keys": diff,
                    }
                )
        lane = script.removesuffix(".py").removesuffix("_real_drill")
        body: dict[str, Any] = {
            "schema": "replay_manifest.v1",
            "kind": "replay_manifest",
            "research_only": True,
            "live_pnl_claim": False,
            "data_label": "yahoo_eod",
            "producer": f"scripts/{script}",
            "replay": {
                "argv": argv,
                "artifacts": artifacts,
                "input_tapes": [{"manifest": TAPE_MANIFEST}],
            },
            "reproduces": reproduces,
        }
        carrier = _seal_carrier(body)
        path = root / "data" / "manifests" / "replay" / f"{lane}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(carrier, indent=2, sort_keys=True) + "\n")
        carriers.append(path.name)

    # serial lane: digest-named multi-file output
    argv = ["scripts/serial_real_drill.py", TAPE_ARG, "--out", REPLAY_OUT]
    proc = subprocess.run(
        [sys.executable, *argv], cwd=root, capture_output=True, text=True, timeout=900
    )
    if proc.returncode == 0:
        artifacts = [
            {
                "path": f"{REPLAY_OUT}/{p.name}",
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted(out_dir.glob("serial_watch_*.json"))
        ]
        body = {
            "schema": "replay_manifest.v1",
            "kind": "replay_manifest",
            "research_only": True,
            "live_pnl_claim": False,
            "data_label": "yahoo_eod",
            "producer": "scripts/serial_real_drill.py",
            "replay": {
                "argv": argv,
                "artifacts": artifacts,
                "input_tapes": [{"manifest": TAPE_MANIFEST}],
            },
        }
        carrier = _seal_carrier(body)
        path = root / "data" / "manifests" / "replay" / "serial.json"
        path.write_text(json.dumps(carrier, indent=2, sort_keys=True) + "\n")
        carriers.append(path.name)
    else:
        failures.append(f"serial: exit {proc.returncode}")

    # monitor / race take argparse --bars/--out
    for script, receipt_name in (
        ("monitor_real_drill.py", "monitor_run_real_drill.json"),
        ("race_real_drill.py", "fleet_race_real_drill.json"),
    ):
        argv = [
            "scripts/" + script,
            "--bars",
            TAPE_ARG,
            "--out",
            f"{REPLAY_OUT}/{receipt_name}",
        ]
        proc = subprocess.run(
            [sys.executable, *argv], cwd=root, capture_output=True, text=True, timeout=1800
        )
        if proc.returncode != 0:
            failures.append(f"{script}: exit {proc.returncode}: {proc.stderr[-300:]}")
            continue
        produced = out_dir / receipt_name
        artifacts = [
            {
                "path": f"{REPLAY_OUT}/{receipt_name}",
                "sha256": hashlib.sha256(produced.read_bytes()).hexdigest(),
            }
        ]
        committed = root / "receipts" / receipt_name
        reproduces = []
        if committed.is_file():
            equal, diff = _claims_equal(
                json.loads(committed.read_text()), json.loads(produced.read_text())
            )
            reproduces.append(
                {
                    "receipt": f"receipts/{receipt_name}",
                    "committed_sha256": hashlib.sha256(committed.read_bytes()).hexdigest(),
                    "claims_equal": equal,
                    "claim_diff_keys": diff,
                }
            )
        lane = script.removesuffix(".py").removesuffix("_real_drill")
        body = {
            "schema": "replay_manifest.v1",
            "kind": "replay_manifest",
            "research_only": True,
            "live_pnl_claim": False,
            "data_label": "yahoo_eod",
            "producer": f"scripts/{script}",
            "replay": {
                "argv": argv,
                "artifacts": artifacts,
                "input_tapes": [{"manifest": TAPE_MANIFEST}],
            },
            "reproduces": reproduces,
        }
        carrier = _seal_carrier(body)
        path = root / "data" / "manifests" / "replay" / f"{lane}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(carrier, indent=2, sort_keys=True) + "\n")
        carriers.append(path.name)

    print(f"carriers written: {len(carriers)}")
    for f in failures:
        print("FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
