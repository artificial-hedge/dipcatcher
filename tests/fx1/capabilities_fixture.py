"""Fixture catalog for ``dipcatcher.capabilities.v1``.

The document is intentionally local to the planner tests. It follows the
producer schema strictly enough that a drift in required fields fails
``CapabilityCatalog.from_json``. Qualification ``ruleset`` and ``audited_on``
are included as optional fields the consumer ignores.
"""

from __future__ import annotations

import hashlib
from typing import Any

from fx1.interactive.contracts import CapabilityCatalog

OVERLAY_SHA256 = hashlib.sha256(b"dipcatcher-capabilities-fixture-v1").hexdigest()
_EVIDENCE = "tests/fx1/capabilities_fixture.py"


def sample_catalog_document() -> dict[str, Any]:
    """Return a fresh, schema-valid catalog document."""
    return {
        "capabilities": [
            _doctor(),
            _verify_research(),
            _verify_receipt(),
            _vol_bench(),
            _calibration(),
            _research(),
            _forecast_eval(),
            _optimize(),
            _train(),
            _alpha_decay(),
        ],
        "generated_by": {"n_lab_commands": 10, "overlay_sha256": OVERLAY_SHA256},
        "schema": "dipcatcher.capabilities.v1",
    }


def sample_catalog() -> CapabilityCatalog:
    """Parse :func:`sample_catalog_document`."""
    return CapabilityCatalog.from_json(sample_catalog_document())


def capability_entry(document: dict[str, Any], capability_id: str) -> dict[str, Any]:
    """Return the raw capability object, raising ``KeyError`` when absent."""
    for entry in document["capabilities"]:
        if entry["id"] == capability_id:
            return entry
    raise KeyError(capability_id)


def _doctor() -> dict[str, Any]:
    return _cap(
        "doctor",
        purpose="Check environment health and the lab manifest.",
        tasks=["environment_health"],
        aliases=["doctor", "health check", "readiness"],
        role="verification",
        read_only=True,
        timeout_s=5,
        evidence_class="VERIFICATION",
        timing="not applicable; local environment probe",
        args=[
            _arg(
                "--strict", "option", "flag", default=False, help="Fail when warnings are present."
            ),
            _arg("subject", "argument", "str", help="Optional subject name."),
        ],
    )


def _verify_research() -> dict[str, Any]:
    return _cap(
        "verify-research",
        purpose="Verify research receipt provenance for a completed run.",
        tasks=["research_provenance"],
        aliases=["verify research", "research provenance"],
        role="verification",
        read_only=True,
        timeout_s=20,
        evidence_class="VERIFICATION",
        timing="after a research run has written a receipt",
        args=[
            _arg("--run-id", "option", "str", help="Run id to verify."),
            _arg("--note-file", "option", "path", path_role="any", help="Optional note file."),
            _arg("--tag", "option", "str", repeatable=True, help="Repeatable tag."),
        ],
    )


def _verify_receipt() -> dict[str, Any]:
    return _cap(
        "verify-receipt",
        purpose="Verify a research receipt digest and schema.",
        tasks=["receipt_digest_check"],
        aliases=["verify receipt", "receipt digest"],
        role="verification",
        read_only=True,
        timeout_s=15,
        evidence_class="VERIFICATION",
        timing="when a receipt file is already on disk",
        args=[_arg("--receipt", "option", "path", path_role="input", help="Receipt path.")],
    )


def _vol_bench() -> dict[str, Any]:
    return _cap(
        "vol-bench",
        purpose="Score volatility forecasts with QLIKE on sealed synthetic shards.",
        tasks=["volatility_forecast_evaluation", "qlike_scoring"],
        aliases=["vol bench", "realized variance", "garch", "har model"],
        role="evaluation",
        read_only=False,
        writes=["receipts"],
        timeout_s=60,
        output_chars=12000,
        evidence_class="SYNTHETIC",
        labels=["SYNTHETIC"],
        timing="not applicable; seeded synthetic shards",
        scores=["QLIKE", "MSE"],
        receipt_schema="vol_bench.v1",
        post_checks=[{"from": "receipt_stdout_marker", "kind": "verify_receipt"}],
        allowed_args={
            "--horizons": {},
            "--models": {"choices": ["har", "garch"]},
            "--n-origins": {"max": 8},
            "--verbose": {},
        },
        args=[
            _arg(
                "--horizons",
                "option",
                "csv",
                default="1,5",
                item_type="int",
                help="Forecast horizons.",
            ),
            _arg(
                "--models",
                "option",
                "csv",
                choices=["har", "garch", "ewma"],
                default="har",
                item_type="str",
                help="Volatility models.",
            ),
            _arg(
                "--n-origins",
                "option",
                "int",
                default=1,
                help="Origin count.",
                max=50,
                min=1,
            ),
            _arg("--verbose", "option", "flag", default=False, help="Verbose score lines."),
        ],
    )


def _calibration() -> dict[str, Any]:
    return _cap(
        "calibration-eval",
        purpose="Evaluate probability calibration with PIT and ECE.",
        tasks=["calibration_evaluation"],
        aliases=["calibration eval", "probability integral", "pit score"],
        role="evaluation",
        read_only=False,
        writes=["receipts"],
        timeout_s=45,
        output_chars=12000,
        evidence_class="SYNTHETIC",
        labels=["SYNTHETIC"],
        timing="not applicable; seeded synthetic shards",
        scores=["PIT", "ECE"],
        receipt_schema="calibration_eval.v1",
        post_checks=[{"kind": "verify_receipt"}],
        args=[
            _arg("--bins", "option", "int", default=10, help="PIT bins.", max=100, min=1),
            _arg(
                "--method",
                "option",
                "enum",
                choices=["histogram", "kernel"],
                default="histogram",
                help="Calibration method.",
            ),
            _arg(
                "--bandwidth",
                "option",
                "float",
                default=0.5,
                help="Kernel bandwidth.",
                max=2,
                min=0.01,
            ),
        ],
    )


def _research() -> dict[str, Any]:
    return _cap(
        "research",
        purpose=(
            "Run the full REQUIRED scorecard and report proper scores. "
            "It cannot be narrowed to one family."
        ),
        tasks=["full_scorecard"],
        aliases=["full scorecard", "benchmark suite", "research scorecard"],
        role="research",
        read_only=False,
        writes=["receipts"],
        timeout_s=600,
        output_chars=20000,
        evidence_class="SYNTHETIC",
        labels=["SYNTHETIC"],
        timing="not applicable; seeded synthetic scorecard",
        scores=["pinball", "CRPS", "QLIKE", "Brier"],
        receipt_schema="research_scorecard.v1",
        dependencies=["doctor"],
        post_checks=[{"kind": "verify_receipt"}],
        args=[
            _arg("--out-dir", "option", "path", path_role="output_dir", help="Receipt directory."),
            _arg("--seed", "option", "int", default=0, help="Sampler seed.", max=10000, min=0),
        ],
    )


def _forecast_eval() -> dict[str, Any]:
    return _cap(
        "forecast-eval",
        purpose="Score a user-supplied forecast column with proper scores.",
        tasks=["user_forecast_scoring"],
        aliases=["forecast eval", "score my forecasts", "user forecast"],
        role="evaluation",
        read_only=False,
        writes=["receipts"],
        timeout_s=90,
        output_chars=12000,
        evidence_class="RESEARCH",
        labels=["USER_SUPPLIED"],
        requires_dataset=True,
        timing="the caller must supply a dataset and forecast column",
        scores=["QLIKE", "pinball"],
        receipt_schema="forecast_eval.v1",
        post_checks=[{"kind": "verify_receipt"}],
        args=[
            _arg(
                "--dataset",
                "option",
                "path",
                help="User dataset.",
                path_role="input",
                required=True,
            ),
            _arg("--target", "option", "str", default="variance", help="Target column."),
            _arg("--horizon", "option", "int", default=1, help="Forecast horizon.", max=64, min=1),
        ],
    )


def _optimize() -> dict[str, Any]:
    return _cap(
        "optimize",
        purpose="Search hyperparameters inside a fixed trial budget.",
        tasks=["hyperparameter_optimization"],
        aliases=["hyperparameter search", "optuna"],
        role="optimization",
        read_only=False,
        consequential=True,
        timeout_s=300,
        output_chars=8000,
        evidence_class="RESEARCH",
        timing="uses the lab default search space",
        args=[
            _arg("--trials", "option", "int", default=5, help="Trial budget.", max=20, min=1),
            _arg("--study", "option", "str", default="default", help="Study name."),
        ],
    )


def _train() -> dict[str, Any]:
    return _cap(
        "train",
        purpose="Fine-tune model weights.",
        tasks=["model_training"],
        aliases=["fine tune", "lora train"],
        role="model_training",
        read_only=False,
        consequential=True,
        writes=["checkpoints"],
        timeout_s=3600,
        executable=False,
        reason="no qualification overlay entry; missing qualification metadata fails closed",
        status="unqualified",
        evidence=[],
        evidence_class="RESEARCH",
        timing="not qualified for execution",
        args=[_arg("--epochs", "option", "int", default=1, help="Epoch cap.", max=5, min=1)],
    )


def _alpha_decay() -> dict[str, Any]:
    return _cap(
        "alpha-decay",
        purpose="Historical alpha-decay diagnostic retained for receipt verification.",
        tasks=["alpha_decay"],
        aliases=["alpha decay"],
        role="evaluation",
        read_only=True,
        timeout_s=10,
        executable=False,
        reason="retired: historical receipts remain verifiable; new selection is blocked",
        status="retired",
        evidence=["historical"],
        evidence_class="RESEARCH",
        timing="retired; historical receipts remain listable",
        args=[_arg("--window", "option", "int", default=20, help="Window length.", max=250, min=1)],
    )


def _cap(
    cap_id: str,
    *,
    purpose: str,
    tasks: list[str],
    aliases: list[str],
    role: str,
    read_only: bool,
    timeout_s: float,
    evidence_class: str,
    timing: str,
    args: list[dict[str, Any]],
    writes: list[str] | None = None,
    consequential: bool = False,
    output_chars: int = 4000,
    labels: list[str] | None = None,
    requires_dataset: bool = False,
    scores: list[str] | None = None,
    receipt_schema: str = "",
    dependencies: list[str] | None = None,
    post_checks: list[dict[str, Any]] | None = None,
    allowed_args: dict[str, Any] | None = None,
    executable: bool = True,
    reason: str = "",
    status: str = "qualified",
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "aliases": aliases,
        "args": args,
        "command": {
            "argv": [cap_id],
            "consequential": consequential,
            "read_only": read_only,
            "role": role,
            "timeout_s": timeout_s,
            "writes": [] if writes is None else writes,
        },
        "dependencies": [] if dependencies is None else dependencies,
        "evidence_class": evidence_class,
        "executable": executable,
        "executable_reason": reason,
        "id": cap_id,
        "inputs": {
            "availability_timing": timing,
            "data_labels": [] if labels is None else labels,
            "requires_dataset": requires_dataset,
        },
        "limits": {"max_output_chars": output_chars, "max_runtime_s": timeout_s},
        "outputs": {
            "receipt_flag": ["--receipt-version", "2"] if writes and "receipts" in writes else [],
            "receipt_schema": receipt_schema,
            "receipt_stdout_marker": "receipt=" if receipt_schema else "",
            "scores": [] if scores is None else scores,
        },
        "post_checks": [] if post_checks is None else post_checks,
        "purpose": purpose,
        "qualification": {
            "audited_on": "2026-10-12",
            "evidence": [_EVIDENCE] if evidence is None else evidence,
            "ruleset": "overlay-v1",
            "status": status,
        },
        "tasks": tasks,
        "version": "1",
    }
    if allowed_args is not None:
        entry["allowed_args"] = allowed_args
    return entry


def _arg(name: str, kind: str, typ: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "help": "Argument.",
        "kind": kind,
        "name": name,
        "required": False,
        "type": typ,
    }
    payload.update(extra)
    return payload
