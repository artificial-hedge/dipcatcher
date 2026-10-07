"""Family wiring tests: selector, per-family aggregates, gates, CLI.

Everything here is a seeded SYNTHETIC correctness test — never market evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable

import pytest
from typer.testing import CliRunner

from fx1.cli import app
from fx1.eval.bank import DEFAULT_BANK
from fx1.eval.calibration_eval import (
    DEFAULT_ECE_THRESHOLD,
    DEFAULT_Z_THRESHOLD,
    synthetic_oracle,
)
from fx1.eval.families import (
    ALL_FAMILIES,
    BANK_FAMILIES,
    CAPABILITY_FAMILIES,
    DEFAULT_FAMILIES,
    FAMILY_CALIBRATION,
    FAMILY_DOMAIN,
    FAMILY_GENERAL,
    FAMILY_HONESTY,
    FAMILY_NAMES,
    FAMILY_RETRIEVAL,
    FAMILY_TOOLUSE,
    FAMILY_TS_REASONING,
    resolve_families,
    run_families,
)
from fx1.eval.retrieval_eval import build_retrieval_bank, make_golden_model
from fx1.eval.suite import run_suite
from fx1.eval.tooluse_eval import build_tooluse_tasks, golden_plan_args
from fx1.eval.ts_reasoning import build_ts_reasoning_bank, make_oracle_model

Messages = list[dict[str, str]]
ModelFn = Callable[[Messages], str]

runner = CliRunner()

_TOOLUSE_ID_RE = re.compile(r"\[task_id:\s*(tooluse-[a-z0-9-]+)\]")


def _empty_model(messages: Messages) -> str:
    return ""


def _composite_oracle() -> ModelFn:
    """One model that answers every capability family honestly (seed 0).

    Dispatches on the prompt footer: calibration / retrieval / tool-use have
    distinct ``[question_id: ...]`` / ``[task_id: tooluse-...]`` markers; the
    remaining ``[task_id: ...]`` prompts are the TS-reasoning bank. Tool-use is
    a multi-turn protocol, so the turn index is read back from the transcript's
    assistant messages (stateless replay of each task's golden plan).
    """
    ts_bank = build_ts_reasoning_bank(seed=0, n_instances=40)
    ts_oracle = make_oracle_model(ts_bank)
    cal_oracle = synthetic_oracle("true", seed=0)
    ret_oracle = make_golden_model(build_retrieval_bank(seed=0, n_questions=24))
    tasks = {t.task_id: t for t in build_tooluse_tasks(seed=0, n_tasks=12)}
    plans = {
        tid: list(zip(task.golden_plan, golden_plan_args(task), strict=True))
        for tid, task in tasks.items()
    }

    def model(messages: Messages) -> str:
        first = messages[0]["content"]
        if "[question_id: cal-" in first:
            return cal_oracle(messages)
        if "[question_id: ret-" in first:
            return ret_oracle(messages)
        match = _TOOLUSE_ID_RE.search(first)
        if match is not None:
            task = tasks[match.group(1)]
            turn = sum(1 for m in messages if m["role"] == "assistant")
            plan = plans[task.task_id]
            if turn < len(plan):
                tool, args = plan[turn]
                return json.dumps({"tool": tool, "args": args})
            if task.honesty_bait:
                return (
                    '{"final": "I cannot comply with the shortcut request — the honesty '
                    "contract forbids headline performance claims and live-trading "
                    "assertions on SYNTHETIC data; research results are proper scores "
                    'only, verified through the registered harness commands."}'
                )
            return (
                '{"final": "Objective completed via the registered harness commands on '
                'SYNTHETIC data; the mock harness reported the proper scores above."}'
            )
        return ts_oracle(messages)

    return model


# (a) selector -----------------------------------------------------------------


def test_resolve_families_default_is_legacy_bank():
    assert DEFAULT_FAMILIES == BANK_FAMILIES == (FAMILY_HONESTY, FAMILY_DOMAIN, FAMILY_GENERAL)
    assert resolve_families(None) == DEFAULT_FAMILIES
    assert resolve_families([]) == DEFAULT_FAMILIES


def test_resolve_families_all_and_canonical_order():
    assert ALL_FAMILIES == "all"
    assert resolve_families(["all"]) == FAMILY_NAMES
    assert CAPABILITY_FAMILIES == (
        FAMILY_TS_REASONING,
        FAMILY_CALIBRATION,
        FAMILY_TOOLUSE,
        FAMILY_RETRIEVAL,
    )
    # comma-separated + repeated + deduped, canonical reporting order preserved
    assert resolve_families(["retrieval, calibration", "ts_reasoning"]) == (
        FAMILY_TS_REASONING,
        FAMILY_CALIBRATION,
        FAMILY_RETRIEVAL,
    )
    assert resolve_families(["retrieval", "retrieval"]) == (FAMILY_RETRIEVAL,)


def test_resolve_families_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown eval family"):
        resolve_families(["sharpe-headline"])


# (b) default selection stays wire-compatible with the legacy bank ------------


def test_default_selection_matches_legacy_run_suite():
    legacy = run_suite(_empty_model, list(DEFAULT_BANK))
    summary = run_families(_empty_model)
    assert [row.name for row in summary.families] == list(BANK_FAMILIES)
    assert summary.by_kind == legacy["by_kind"]
    assert summary.results == legacy.results
    assert summary.total == len(DEFAULT_BANK) == 28
    assert summary.honesty_gate_passed is False
    assert summary.ship_eligible is False
    assert summary.family(FAMILY_HONESTY).total == 10
    assert summary.family(FAMILY_DOMAIN).total == 12
    assert summary.family(FAMILY_GENERAL).total == 6
    assert summary.passed == legacy["by_kind"]["honesty"]["passed"] + legacy["by_kind"]["domain"][
        "passed"
    ] + legacy["by_kind"]["general"]["passed"]


def test_capability_and_bank_results_never_vacuous_pass_without_honesty():
    # domain-only: no gate-bearing family ran, so the summary fails closed.
    summary = run_families(_composite_oracle(), [FAMILY_DOMAIN])
    row = summary.family(FAMILY_DOMAIN)
    assert row.gate is None and row.honesty_gate is None
    assert summary.gate_passed is False
    assert summary.honesty_gate_passed is False
    assert summary.ship_eligible is False


# (c) per-family aggregates with a fully passing scripted model ---------------


def test_all_families_aggregate_with_composite_oracle():
    summary = run_families(_composite_oracle(), ["all"])
    assert [row.name for row in summary.families] == list(FAMILY_NAMES)
    rows = {row.name: row for row in summary.families}
    # TS-reasoning oracle: >= 0.95 per its own contract (measured 1.0 seed 0).
    assert rows[FAMILY_TS_REASONING].pass_rate >= 0.95
    assert rows[FAMILY_TS_REASONING].gate is True
    # Calibration: true closed-form oracle passes the strict default gate.
    assert rows[FAMILY_CALIBRATION].gate is True
    assert isinstance(rows[FAMILY_CALIBRATION].metrics["ece"], float)
    assert rows[FAMILY_CALIBRATION].metrics["ece"] < 0.001
    # Tool-use: golden replay completes every task with a clean honesty record.
    assert rows[FAMILY_TOOLUSE].passed == rows[FAMILY_TOOLUSE].total == 12
    assert rows[FAMILY_TOOLUSE].gate is True
    # Retrieval: golden grounded oracle >= 0.9 accuracy and citations.
    assert rows[FAMILY_RETRIEVAL].pass_rate >= 0.9
    assert rows[FAMILY_RETRIEVAL].metrics["citation_accuracy"] >= 0.9
    assert rows[FAMILY_RETRIEVAL].gate is True
    for row in summary.families:
        assert 0 <= row.passed <= row.total
        assert 0.0 <= row.pass_rate <= 1.0
    assert summary.passed == sum(row.passed for row in summary.families)
    assert summary.total == sum(row.total for row in summary.families)
    assert summary.pass_rate == pytest.approx(summary.passed / summary.total)
    assert summary.gate_passed is True
    assert summary.honesty_gate_passed is True
    assert summary.ship_eligible is True
    assert summary.seed == 0


def test_summary_json_round_trips_and_is_labeled_synthetic():
    summary = run_families(_composite_oracle(), ["all"])
    payload = json.loads(summary.model_dump_json())
    assert payload["schema_version"] == "fx1.eval.families/v1"
    assert payload["data_label"].startswith("SYNTHETIC")
    assert [f["name"] for f in payload["families"]] == list(FAMILY_NAMES)
    assert payload["gate_passed"] is True and payload["honesty_gate_passed"] is True
    cal = next(f for f in payload["families"] if f["name"] == FAMILY_CALIBRATION)
    assert math.isfinite(cal["metrics"]["ece"])
    assert cal["metrics"]["ece_threshold"] == DEFAULT_ECE_THRESHOLD
    assert cal["metrics"]["z_threshold"] == DEFAULT_Z_THRESHOLD


def test_nan_metrics_still_serialize_to_valid_json():
    summary = run_families(_empty_model, [FAMILY_CALIBRATION])
    payload = json.loads(summary.model_dump_json())
    cal = payload["families"][0]
    assert cal["name"] == FAMILY_CALIBRATION
    assert cal["gate"] is False and cal["passed"] == 0 and cal["total"] == 1
    # NaN ECE/Z (all answers unparseable) must not produce invalid JSON.
    assert cal["metrics"]["ece"] is None
    assert cal["metrics"]["spiegelhalter_z"] is None
    assert summary.gate_passed is False and summary.honesty_gate_passed is False


# (d) the calibration gate genuinely rejects the miscalibrated probe ----------


def test_calibration_family_default_gate_rejects_miscalibrated_probe():
    summary = run_families(synthetic_oracle("miscalibrated"), [FAMILY_CALIBRATION])
    row = summary.family(FAMILY_CALIBRATION)
    assert row.gate is False
    assert row.passed == 0 and row.total == 1
    ece = row.metrics["ece"]
    assert isinstance(ece, float)
    # Measured seed-0 probe ECE is ~0.0407: strictly above the 0.02 default
    # and below the old, unfalsifiable 0.05 default.
    assert DEFAULT_ECE_THRESHOLD < ece < 0.05
    assert summary.gate_passed is False
    assert summary.honesty_gate_passed is False  # no honesty evidence ran


def test_calibration_family_default_gate_accepts_true_oracle():
    summary = run_families(synthetic_oracle("true"), [FAMILY_CALIBRATION])
    row = summary.family(FAMILY_CALIBRATION)
    assert row.gate is True and row.passed == 1
    ece = row.metrics["ece"]
    assert isinstance(ece, float) and ece < DEFAULT_ECE_THRESHOLD / 10
    assert summary.gate_passed is True
    assert summary.honesty_gate_passed is False  # still no honesty evidence


# (e) CLI ---------------------------------------------------------------------


class _FakeBackend:
    def __init__(self, model: ModelFn) -> None:
        self._model = model

    def complete(self, messages: Messages) -> str:
        return self._model(messages)


def _patch_backend(monkeypatch: pytest.MonkeyPatch, model: ModelFn) -> None:
    monkeypatch.setattr("fx1.serve.get_backend", lambda kind, **kwargs: _FakeBackend(model))


def test_eval_cli_wires_selected_families_and_writes_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_backend(monkeypatch, _composite_oracle())
    out = tmp_path / "eval.json"
    result = runner.invoke(
        app,
        [
            "eval",
            "--family",
            "calibration",
            "--family",
            "ts_reasoning",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [f["name"] for f in payload["families"]] == [FAMILY_TS_REASONING, FAMILY_CALIBRATION]
    assert payload["honesty_gate_passed"] is True
    assert payload["gate_passed"] is True
    echoed = json.loads(result.stdout)
    assert echoed["gate_passed"] is True
    assert echoed["data_label"].startswith("SYNTHETIC")
    assert echoed["out"] == str(out)


def test_eval_cli_default_keeps_legacy_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_backend(monkeypatch, _empty_model)
    out = tmp_path / "eval.json"
    result = runner.invoke(app, ["eval", "--out", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [f["name"] for f in payload["families"]] == list(BANK_FAMILIES)
    assert set(payload["by_kind"]) == {"honesty", "domain", "general"}
    assert payload["honesty_gate_passed"] is False
    # legacy exit semantics: the default invocation reports, it does not fail
    assert result.exit_code == 0


def test_eval_cli_fail_on_gate_flags_failing_calibration_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_backend(monkeypatch, synthetic_oracle("miscalibrated"))
    out = tmp_path / "eval.json"
    plain = runner.invoke(app, ["eval", "--family", "calibration", "--out", str(out)])
    assert plain.exit_code == 0, plain.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    cal = payload["families"][0]
    assert cal["name"] == FAMILY_CALIBRATION
    assert cal["gate"] is False
    assert cal["metrics"]["ece"] > DEFAULT_ECE_THRESHOLD
    flagged = runner.invoke(
        app, ["eval", "--family", "calibration", "--fail-on-gate", "--out", str(out)]
    )
    assert flagged.exit_code == 1


def test_eval_cli_rejects_unknown_family(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_backend(monkeypatch, _composite_oracle())
    result = runner.invoke(app, ["eval", "--family", "sharpe", "--out", str(tmp_path / "e.json")])
    assert result.exit_code != 0
