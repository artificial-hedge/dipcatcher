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

Usage: ``PYTHONPATH=src python scripts/gen_drill_replay_manifests.py [lane ...]``
runs every drill under ``--out data/metadata/replay`` and seals carriers
into ``data/manifests/replay/<lane>.json`` — kept out of ``receipts/`` so
CI (no tape materialized) never picks them up as replayable receipts and
the corpus audit never treats them as inputs.
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

# Fully synthetic drills — no input tape, replayable anywhere (CI included).
# argv is the full argv; out stays under REPLAY_OUT so artifacts never land
# on a committed path (the replay committed-overwrite guard would fire).
SYNTHETIC_DRILLS: dict[str, dict[str, Any]] = {
    "mcs_vol_drill.py": {
        "lane": "mcs_vol",
        "argv_extra": ["--out-dir", REPLAY_OUT],
        "receipts": ["mcs_vol_drill.json"],
    },
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


def _drift_reason(committed: dict[str, Any], produced: dict[str, Any], diff: list[str]) -> str:
    """Classify why a produced artifact's claims differ from the committed one."""
    left, right = _norm(committed), _norm(produced)
    added = [k for k in diff if k not in left and k in right]
    dropped = [k for k in diff if k in left and k not in right]
    if added and not dropped:
        return "contract_field_addition"
    if set(diff) == {"n_models"} or any(k in diff for k in ("heads", "fleet_heads", "registry")):
        return "registry_growth"
    for key in diff:
        lv, rv = left.get(key), right.get(key)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)) and rv > lv:
            return "registry_growth"
    return "lane_evolution"


def _reproduces_entry(receipt: str, committed_path: Path, produced_path: Path) -> dict[str, Any]:
    committed = json.loads(committed_path.read_text())
    produced = json.loads(produced_path.read_text())
    equal, diff = _claims_equal(committed, produced)
    entry: dict[str, Any] = {
        "receipt": receipt,
        "committed_sha256": hashlib.sha256(committed_path.read_bytes()).hexdigest(),
        "claims_equal": equal,
        "claim_diff_keys": diff,
    }
    if not equal:
        entry["drift_reason"] = _drift_reason(committed, produced, diff)
    return entry


def _seal_carrier(body: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(body)
    sealed["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return sealed


def main() -> int:
    only = {a for a in sys.argv[1:] if not a.startswith("--")}
    root = Path.cwd()
    out_dir = root / REPLAY_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    us_wide = root / TAPE_MANIFEST
    if not us_wide.is_file():
        raise SystemExit(f"tape manifest missing: {us_wide}")

    carriers: list[str] = []
    failures: list[str] = []

    for script, spec in DRILLS.items():
        lane_name = script.removesuffix(".py").removesuffix("_real_drill").removesuffix("_drill")
        if only and lane_name not in only and script not in only:
            continue
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
                reproduces.append(_reproduces_entry(f"receipts/{name}", committed, produced))
        lane = script.removesuffix(".py").removesuffix("_real_drill").removesuffix("_drill")
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
    if not only or "serial" in only or "serial_real_drill.py" in only:
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
        lane_name = script.removesuffix(".py").removesuffix("_real_drill").removesuffix("_drill")
        if only and lane_name not in only and script not in only:
            continue
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
            reproduces.append(_reproduces_entry(f"receipts/{receipt_name}", committed, produced))
        lane = script.removesuffix(".py").removesuffix("_real_drill").removesuffix("_drill")
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

    # corpus lane: a frozen-membership audit. The membership file plus
    # every member receipt are pinned as input tapes, so the BH-FDR/
    # e-value pool is byte-reproducible even as the live corpus grows —
    # new receipts simply are not members of this snapshot.
    if not only or "corpus" in only or "corpus_real_drill.py" in only:
        membership_rel = "data/manifests/replay_corpus_members.json"
        members = json.loads((root / membership_rel).read_text())
        input_tapes = [
            {
                "path": f"receipts/{name}",
                "sha256": hashlib.sha256((root / "receipts" / name).read_bytes()).hexdigest(),
            }
            for name in members
        ]
        input_tapes.append(
            {
                "path": membership_rel,
                "sha256": hashlib.sha256((root / membership_rel).read_bytes()).hexdigest(),
            }
        )
        argv = [
            "scripts/corpus_real_drill.py",
            "--receipts",
            "receipts",
            "--membership",
            membership_rel,
            "--out",
            f"{REPLAY_OUT}/corpus_real_drill.json",
        ]
        proc = subprocess.run(
            [sys.executable, *argv], cwd=root, capture_output=True, text=True, timeout=900
        )
        if proc.returncode != 0:
            failures.append(f"corpus: exit {proc.returncode}: {proc.stderr[-300:]}")
        else:
            produced = out_dir / "corpus_real_drill.json"
            artifacts = [
                {
                    "path": f"{REPLAY_OUT}/corpus_real_drill.json",
                    "sha256": hashlib.sha256(produced.read_bytes()).hexdigest(),
                }
            ]
            committed = root / "receipts" / "corpus_real_drill.json"
            reproduces = []
            if committed.is_file():
                reproduces.append(
                    _reproduces_entry("receipts/corpus_real_drill.json", committed, produced)
                )
            body = {
                "schema": "replay_manifest.v1",
                "kind": "replay_manifest",
                "research_only": True,
                "live_pnl_claim": False,
                "data_label": "MIXED",
                "producer": "scripts/corpus_real_drill.py",
                "replay": {"argv": argv, "artifacts": artifacts, "input_tapes": input_tapes},
                "reproduces": reproduces,
            }
            carrier = _seal_carrier(body)
            path = root / "data" / "manifests" / "replay" / "corpus.json"
            path.write_text(json.dumps(carrier, indent=2, sort_keys=True) + "\n")
            carriers.append(path.name)

    # synthetic lanes: no input tape, replayable anywhere
    for script, spec in SYNTHETIC_DRILLS.items():
        lane = spec["lane"]
        if only and lane not in only and script not in only:
            continue
        argv = ["scripts/" + script, *spec["argv_extra"]]
        proc = subprocess.run(
            [sys.executable, *argv], cwd=root, capture_output=True, text=True, timeout=1800
        )
        if proc.returncode != 0:
            failures.append(f"{script}: exit {proc.returncode}: {proc.stderr[-300:]}")
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
                reproduces.append(_reproduces_entry(f"receipts/{name}", committed, produced))
        body = {
            "schema": "replay_manifest.v1",
            "kind": "replay_manifest",
            "research_only": True,
            "live_pnl_claim": False,
            "data_label": "SYNTHETIC",
            "producer": f"scripts/{script}",
            "replay": {"argv": argv, "artifacts": artifacts},
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
