"""Curated qualification overlay for the lab capability contract.

Capabilities absent from ``CURATED_OVERLAY`` stay unqualified: a missing audit
is not permission to execute. ``audited_on`` is the overlay revision date, not
a clock reading. The sha256 is taken over canonical JSON (sorted keys, compact
separators) so the digest does not depend on dict insertion order.

``timeout_s`` / ``limits`` are harness ceilings the caller must enforce. They
are not measured runtimes. The 2026-10-12 revision records selective family
runs (``research-only``) and refuses commands whose default output is ``receipts/``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict

SCHEMA = "dipcatcher.capabilities.v1"
RULESET = "overlay-v1"
AUDITED_ON = "2026-10-12"
FAIL_CLOSED_REASON = "no qualification overlay entry; missing qualification metadata fails closed"
RESEARCH_ONLY_ID = "research-only"
PARTIAL_RUN_SCHEMA = "dipcatcher.family_run.v1"
VERIFY_ALL_UNQUALIFIED_REASON = (
    "writes a sealed evidence_audit receipt into receipts/ by default; "
    "model-driven execution is refused because that directory is implicit"
)
EVIDENCE_CLASSES = frozenset({"SYNTHETIC", "RESEARCH", "VERIFICATION"})
QUALIFICATION_STATUSES = frozenset({"qualified", "experimental", "unqualified", "retired"})


class ArgHint(TypedDict, total=False):
    """Per-argument tightening applied on top of Click introspection."""

    type: str
    item_type: str
    min: int
    max: int
    choices: list[str]
    default: str | None
    path_role: str
    required: bool


class CommandContract(TypedDict):
    argv: list[str]
    role: str
    read_only: bool
    writes: list[str]
    consequential: bool
    timeout_s: int


class InputContract(TypedDict):
    data_labels: list[str]
    requires_dataset: bool
    availability_timing: str


class OutputContract(TypedDict):
    scores: list[str]
    receipt_schema: str
    receipt_flag: list[str]
    receipt_stdout_marker: str


class LimitContract(TypedDict):
    max_runtime_s: int
    max_output_chars: int


class Qualification(TypedDict):
    status: str
    evidence: list[str]
    ruleset: str
    audited_on: str
    reason: NotRequired[str]


class OverlayEntry(TypedDict):
    id: str
    version: str
    purpose: str
    tasks: list[str]
    aliases: list[str]
    command: CommandContract
    allowed_args: list[str]
    arg_hints: dict[str, ArgHint]
    inputs: InputContract
    outputs: OutputContract
    dependencies: list[str]
    post_checks: list[dict[str, str]]
    limits: LimitContract
    qualification: Qualification
    evidence_class: str


def _csv(item_type: str) -> ArgHint:
    return {"type": "csv", "item_type": item_type}


def _v2() -> ArgHint:
    return {"default": "2", "choices": ["2"], "min": 2, "max": 2}


def _limits(runtime_s: int, output_chars: int = 12000) -> LimitContract:
    return {"max_runtime_s": runtime_s, "max_output_chars": output_chars}


def _qualified(evidence: list[str]) -> Qualification:
    return {
        "status": "qualified",
        "evidence": evidence,
        "ruleset": RULESET,
        "audited_on": AUDITED_ON,
    }


def _command(
    argv: list[str],
    role: str,
    *,
    read_only: bool,
    writes: list[str],
    timeout_s: int,
) -> CommandContract:
    return {
        "argv": argv,
        "role": role,
        "read_only": read_only,
        "writes": list(writes),
        "consequential": False,
        "timeout_s": timeout_s,
    }


def _inputs(labels: list[str], timing: str) -> InputContract:
    return {
        "data_labels": list(labels),
        "requires_dataset": False,
        "availability_timing": timing,
    }


def _receipt_outputs(scores: list[str]) -> OutputContract:
    return {
        "scores": list(scores),
        "receipt_schema": "receipt.v2",
        "receipt_flag": ["--receipt-version", "2"],
        "receipt_stdout_marker": "receipt=",
    }


def _verify_outputs() -> OutputContract:
    return {
        "scores": [],
        "receipt_schema": "",
        "receipt_flag": [],
        "receipt_stdout_marker": "",
    }


def _out_dir() -> ArgHint:
    """Caller-supplied receipt directory. No ``receipts/`` default."""
    return {"path_role": "output_dir", "default": None, "required": True}


def _unqualified(evidence: list[str], reason: str) -> Qualification:
    return {
        "status": "unqualified",
        "evidence": evidence,
        "ruleset": RULESET,
        "audited_on": AUDITED_ON,
        "reason": reason,
    }


def _partial_outputs() -> OutputContract:
    return {
        "scores": [],
        "receipt_schema": PARTIAL_RUN_SCHEMA,
        "receipt_flag": [],
        "receipt_stdout_marker": "json=",
    }


_EXIT: dict[str, str] = {"kind": "exit_code"}
_RECEIPT_CHECK: dict[str, str] = {
    "kind": "verify_receipt",
    "from": "receipt_stdout_marker",
}


def overlay_sha256(entries: Sequence[Mapping[str, Any]] | None = None) -> str:
    """Sha256 of the canonical overlay JSON. Defaults to the curated overlay."""
    payload: Sequence[Mapping[str, Any]] = CURATED_OVERLAY if entries is None else entries
    raw = json.dumps(
        list(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def qualified_ids() -> tuple[str, ...]:
    """Capability ids the curated overlay marks qualified, in overlay order."""
    return tuple(
        entry["id"] for entry in CURATED_OVERLAY if entry["qualification"]["status"] == "qualified"
    )


CURATED_OVERLAY: tuple[OverlayEntry, ...] = (
    {
        "id": "doctor",
        "version": "2",
        "purpose": (
            "Report local research-tree health and write resolved_config.json under "
            "the configured data root. Also ensures that root's raw, bronze, silver, "
            "gold, and metadata directories exist. Does not write receipts/."
        ),
        "tasks": ["lab_health", "preflight"],
        "aliases": ["doctor", "health", "preflight"],
        "command": _command(
            ["doctor"],
            "verification",
            read_only=False,
            writes=["resolved_config", "data_dirs"],
            timeout_s=60,
        ),
        "allowed_args": ["--config"],
        "arg_hints": {},
        "inputs": _inputs([], "n/a — local tree health check; no market dataset is fetched"),
        "outputs": _verify_outputs(),
        "dependencies": [],
        "post_checks": [_EXIT],
        "limits": _limits(60),
        "qualification": _qualified(
            [
                "tests/unit/cli/test_cli_api.py",
                "tests/unit/cli/test_capabilities_cmd.py",
            ]
        ),
        "evidence_class": "VERIFICATION",
    },
    {
        "id": "verify-research",
        "version": "1",
        "purpose": (
            "Verify a research notebook, phase-1 run directory, or evidence index "
            "and exit non-zero when the artifact is invalid."
        ),
        "tasks": ["notebook_verification", "receipt_verification"],
        "aliases": ["verify research", "check notebook"],
        "command": _command(
            ["verify-research"],
            "verification",
            read_only=True,
            writes=[],
            timeout_s=120,
        ),
        "allowed_args": ["path"],
        "arg_hints": {},
        "inputs": _inputs([], "caller-supplied artifact path; nothing is fetched"),
        "outputs": _verify_outputs(),
        "dependencies": [],
        "post_checks": [_EXIT],
        "limits": _limits(120),
        "qualification": _qualified(
            [
                "tests/unit/pipeline/test_phase1_receipt_verification.py",
                "tests/unit/cli/test_cli_main_cov.py",
            ]
        ),
        "evidence_class": "VERIFICATION",
    },
    {
        "id": "verify-receipt",
        "version": "1",
        "purpose": (
            "Verify one sealed receipt file for structure and hash consistency "
            "and exit non-zero on any violation."
        ),
        "tasks": ["receipt_verification"],
        "aliases": ["verify receipt", "check seal"],
        "command": _command(
            ["verify-receipt"],
            "verification",
            read_only=True,
            writes=[],
            timeout_s=60,
        ),
        "allowed_args": ["path"],
        "arg_hints": {},
        "inputs": _inputs([], "caller-supplied receipt path; nothing is fetched"),
        "outputs": _verify_outputs(),
        "dependencies": [],
        "post_checks": [_EXIT],
        "limits": _limits(60),
        "qualification": _qualified(
            [
                "tests/unit/cli/test_cli_lanes.py",
                "tests/unit/research/test_receipt_v2.py",
            ]
        ),
        "evidence_class": "VERIFICATION",
    },
    {
        "id": "verify-all",
        "version": "2",
        "purpose": (
            "Audit a receipts directory for seals, filename binding, duplicate seals, "
            "and evidence-index freshness, and write an evidence-audit receipt. "
            "Not executable: the default output directory is receipts/."
        ),
        "tasks": ["evidence_audit", "receipt_verification"],
        "aliases": ["verify all", "evidence audit"],
        "command": _command(
            ["verify-all"],
            "verification",
            read_only=False,
            writes=["receipts"],
            timeout_s=300,
        ),
        "allowed_args": ["--receipts-dir", "--out-dir"],
        "arg_hints": {
            "--out-dir": {"path_role": "output_dir", "default": None},
            "--receipts-dir": {"path_role": "input", "default": None},
        },
        "inputs": _inputs([], "caller-supplied receipts directory; nothing is fetched"),
        "outputs": {
            "scores": [],
            "receipt_schema": "receipt.v2",
            "receipt_flag": [],
            "receipt_stdout_marker": "receipt=",
        },
        "dependencies": [],
        "post_checks": [_RECEIPT_CHECK],
        "limits": _limits(300),
        "qualification": _unqualified(
            ["tests/unit/research/test_evidence_audit.py"],
            VERIFY_ALL_UNQUALIFIED_REASON,
        ),
        "evidence_class": "VERIFICATION",
    },
    {
        "id": "vol-bench",
        "version": "2",
        "purpose": "Score SYNTHETIC volatility forecasts with QLIKE and MSE and write a sealed receipt.",
        "tasks": ["volatility_forecast_evaluation", "qlike_scoring"],
        "aliases": ["realized variance", "har", "garch", "volatility"],
        "command": _command(
            ["vol-bench"],
            "evaluation",
            read_only=False,
            writes=["receipts", "resolved_config"],
            timeout_s=900,
        ),
        "allowed_args": [
            "--config",
            "--models",
            "--shards",
            "--horizons",
            "--min-history",
            "--n-origins",
            "--stride",
            "--n-bars",
            "--seed",
            "--out-dir",
            "--receipt-version",
        ],
        "arg_hints": {
            "--models": _csv("str"),
            "--shards": _csv("str"),
            "--horizons": _csv("int"),
            "--min-history": {"min": 32, "max": 400},
            "--n-origins": {"min": 10, "max": 24},
            "--stride": {"min": 1, "max": 50},
            "--n-bars": {"min": 1, "max": 2000},
            "--receipt-version": _v2(),
            "--out-dir": _out_dir(),
        },
        "inputs": _inputs(["SYNTHETIC"], "n/a — seeded synthetic shards"),
        "outputs": _receipt_outputs(["QLIKE", "MSE"]),
        "dependencies": [],
        "post_checks": [_RECEIPT_CHECK],
        "limits": _limits(900),
        "qualification": _qualified(
            [
                "tests/unit/research/test_vol_bench.py",
                "tests/unit/cli/test_cli_lanes.py",
            ]
        ),
        "evidence_class": "SYNTHETIC",
    },
    {
        "id": "calibration-eval",
        "version": "2",
        "purpose": (
            "Score SYNTHETIC distribution-fleet calibration with PIT histograms "
            "and reliability diagnostics and write a sealed receipt."
        ),
        "tasks": ["calibration_evaluation", "pit_scoring"],
        "aliases": ["calibration", "pit", "reliability"],
        "command": _command(
            ["calibration-eval"],
            "evaluation",
            read_only=False,
            writes=["receipts", "resolved_config"],
            timeout_s=900,
        ),
        "allowed_args": [
            "--config",
            "--models",
            "--shards",
            "--n-train",
            "--n-eval",
            "--pit-bins",
            "--seed",
            "--out-dir",
            "--receipt-version",
        ],
        "arg_hints": {
            "--models": _csv("str"),
            "--shards": _csv("str"),
            "--n-train": {"min": 32, "max": 512},
            "--n-eval": {"min": 64, "max": 256},
            "--pit-bins": {"min": 5, "max": 50},
            "--receipt-version": _v2(),
            "--out-dir": _out_dir(),
        },
        "inputs": _inputs(["SYNTHETIC"], "n/a — seeded synthetic shards"),
        "outputs": _receipt_outputs(["PIT"]),
        "dependencies": [],
        "post_checks": [_RECEIPT_CHECK],
        "limits": _limits(900),
        "qualification": _qualified(["tests/unit/research/test_calibration_eval.py"]),
        "evidence_class": "SYNTHETIC",
    },
    {
        "id": "rankic",
        "version": "2",
        "purpose": (
            "Score SYNTHETIC cross-sectional rank-IC of challenger transforms "
            "and write a sealed receipt."
        ),
        "tasks": ["rank_ic_evaluation"],
        "aliases": ["rankic", "rank ic", "cross section"],
        "command": _command(
            ["rankic"],
            "evaluation",
            read_only=False,
            writes=["receipts"],
            timeout_s=900,
        ),
        "allowed_args": [
            "--panels",
            "--challengers",
            "--n-assets",
            "--n-dates",
            "--horizons",
            "--seed",
            "--out-dir",
            "--receipt-version",
        ],
        "arg_hints": {
            "--panels": _csv("str"),
            "--challengers": _csv("str"),
            "--horizons": _csv("int"),
            "--n-assets": {"min": 2, "max": 32},
            "--n-dates": {"min": 20, "max": 96},
            "--receipt-version": _v2(),
            "--out-dir": _out_dir(),
        },
        "inputs": _inputs(["SYNTHETIC"], "n/a — seeded synthetic panels"),
        "outputs": _receipt_outputs(["RankIC"]),
        "dependencies": [],
        "post_checks": [_RECEIPT_CHECK],
        "limits": _limits(900),
        "qualification": _qualified(
            [
                "tests/unit/cli/test_cli_lanes.py",
                "tests/property/test_cross_sectional_rankic.py",
            ]
        ),
        "evidence_class": "SYNTHETIC",
    },
    {
        "id": "serial-watch",
        "version": "2",
        "purpose": (
            "Audit SYNTHETIC PIT streams for serial independence with anytime-valid "
            "e-processes and write a sealed receipt."
        ),
        "tasks": ["pit_serial_independence"],
        "aliases": ["serial watch", "pit serial", "serial independence"],
        "command": _command(
            ["serial-watch"],
            "evaluation",
            read_only=False,
            writes=["receipts"],
            timeout_s=300,
        ),
        "allowed_args": [
            "pits",
            "--n-lags",
            "--alpha",
            "--lam",
            "--data-label",
            "--out-dir",
            "--receipt-version",
        ],
        "arg_hints": {
            "--n-lags": {"min": 1, "max": 5},
            "--data-label": {"default": "SYNTHETIC", "choices": ["SYNTHETIC"]},
            "--receipt-version": _v2(),
            "--out-dir": _out_dir(),
        },
        "inputs": _inputs(
            ["SYNTHETIC"],
            "caller-supplied PIT file; pass --data-label SYNTHETIC (CLI default UNKNOWN is not qualified)",
        ),
        "outputs": _receipt_outputs(["e-value"]),
        "dependencies": [],
        "post_checks": [_RECEIPT_CHECK],
        "limits": _limits(300),
        "qualification": _qualified(["tests/unit/research/test_serial_watch.py"]),
        "evidence_class": "SYNTHETIC",
    },
    {
        "id": "lane-power",
        "version": "2",
        "purpose": (
            "Measure SYNTHETIC alarm rate and time of sequential monitor lanes "
            "under injected defects and write a sealed receipt."
        ),
        "tasks": ["sequential_power"],
        "aliases": ["lane power", "alarm rate", "sequential power"],
        "command": _command(
            ["lane-power"],
            "evaluation",
            read_only=False,
            writes=["receipts"],
            timeout_s=900,
        ),
        "allowed_args": [
            "--defects",
            "--n-steps",
            "--n-seeds",
            "--alpha",
            "--lanes",
            "--out-dir",
            "--receipt-version",
        ],
        "arg_hints": {
            "--defects": _csv("float"),
            "--lanes": _csv("str"),
            "--n-steps": {"min": 16, "max": 200},
            "--n-seeds": {"min": 1, "max": 10},
            "--receipt-version": _v2(),
            "--out-dir": _out_dir(),
        },
        "inputs": _inputs(["SYNTHETIC"], "n/a — seeded synthetic streams"),
        "outputs": _receipt_outputs([]),
        "dependencies": [],
        "post_checks": [_RECEIPT_CHECK],
        "limits": _limits(900),
        "qualification": _qualified(["tests/unit/research/test_lane_power.py"]),
        "evidence_class": "SYNTHETIC",
    },
    {
        "id": "research",
        "version": "2",
        "purpose": (
            "Execute the whole REQUIRED research scorecard on the default SYNTHETIC "
            "config and write the labeled notebook. This is the heavy run used when "
            "no family selection is supplied. Selective runs are research-only."
        ),
        "tasks": ["required_scorecard"],
        "aliases": ["research", "scorecard", "full scorecard"],
        "command": _command(
            ["research"],
            "research",
            read_only=False,
            writes=["research_artifacts", "resolved_config"],
            timeout_s=3600,
        ),
        "allowed_args": ["--config"],
        "arg_hints": {},
        "inputs": _inputs(
            ["SYNTHETIC"],
            "n/a — default config runs seeded SYNTHETIC shards; this capability does not accept --only",
        ),
        "outputs": {
            "scores": ["QLIKE"],
            "receipt_schema": "",
            "receipt_flag": [],
            "receipt_stdout_marker": "json=",
        },
        "dependencies": [],
        "post_checks": [{"kind": "verify_research", "from": "json="}],
        "limits": _limits(3600, 48000),
        "qualification": _qualified(["tests/unit/cli/test_cli_main_cov.py"]),
        "evidence_class": "SYNTHETIC",
    },
    {
        "id": RESEARCH_ONLY_ID,
        "version": "1",
        "purpose": (
            "Run only the qualified families named by --only and write a partial "
            "dipcatcher.family_run.v1 record. The record is not a sealed receipt, "
            "verified stays false, and verify-research does not apply."
        ),
        "tasks": ["selective_family_run"],
        "aliases": ["research only", "selective families", "family run"],
        "command": _command(
            ["research"],
            "research",
            read_only=False,
            writes=["family_runs", "resolved_config"],
            timeout_s=3600,
        ),
        "allowed_args": ["--config", "--only"],
        "arg_hints": {"--only": {"type": "csv", "item_type": "str", "required": True}},
        "inputs": _inputs(
            ["SYNTHETIC"],
            "n/a — qualified path uses the default SYNTHETIC config; --only selects runners",
        ),
        "outputs": _partial_outputs(),
        "dependencies": [],
        "post_checks": [],
        "limits": _limits(3600, 48000),
        "qualification": _qualified(
            [
                "tests/unit/research/test_family_surface.py",
                "tests/unit/cli/test_family_cmds.py",
            ]
        ),
        "evidence_class": "SYNTHETIC",
    },
)
