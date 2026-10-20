"""Session grants for consequential harness steps are bound to the argv that runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from fx1.interactive.approvals import ApprovalDecision, ApprovalGate
from fx1.interactive.jobs import JobLedger
from fx1.interactive.planner import make_step_key
from fx1.interactive.runctx import new_run_id, new_session_id
from fx1.interactive.superpower import (
    PlannedCapability,
    SuperpowerContext,
    SuperpowerPlan,
    plan_goal,
    run_plan,
)
from tests.fx1.capabilities_fixture import sample_catalog

_DENIAL = "operator did not approve this consequential command"


@pytest.fixture(autouse=True)
def _fx1_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))


class _Rec:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str] | None]] = []

    def run(self, name: str, extra: list[str] | None = None, **_kwargs: Any) -> SimpleNamespace:
        self.calls.append((name, extra))
        return SimpleNamespace(
            exit_code=0,
            stdout="DATA_LABEL=SYNTHETIC\nSYNTHETIC\nok\n",
            stderr="",
            timed_out=False,
            cancelled=False,
            truncated=False,
        )


def _optimize(trials: int) -> SuperpowerPlan:
    plan = plan_goal("run a hyperparameter search", catalog=sample_catalog())
    step = plan.typed_steps[0]
    args = dict(step.args)
    args["--trials"] = trials
    frozen = MappingProxyType(args)
    updated = replace(step, args=frozen, step_key=make_step_key(step.capability_id, frozen))
    plan.typed_steps = (updated,)
    return plan


def _legacy_optimize() -> SuperpowerPlan:
    return SuperpowerPlan(
        goal="run a hyperparameter search",
        capabilities=[PlannedCapability("optimize", "search")],
    )


def _runs(result: Any) -> dict[str, Any]:
    return {run.id: run for run in result.runs}


def _run(plan: SuperpowerPlan, root: Path, harness: _Rec, gate: ApprovalGate, **extra: Any) -> Any:
    return run_plan(
        plan,
        SuperpowerContext(
            harness=harness,
            run_dir=root,
            approvals=gate,
            catalog=sample_catalog(),
            **extra,
        ),
    )


def test_session_grant_does_not_cover_a_different_argv(tmp_path: Path) -> None:
    prompts = {"n": 0}

    def decide(_request: Any) -> ApprovalDecision:
        prompts["n"] += 1
        return ApprovalDecision.SESSION

    gate = ApprovalGate(decide_fn=decide)
    harness = _Rec()
    first = _run(_optimize(5), tmp_path / "a", harness, gate)
    assert _runs(first)["optimize"].status == "ran"
    assert prompts["n"] == 1
    assert gate.records[0].key == "harness:optimize"
    repeated = _run(_optimize(5), tmp_path / "b", harness, gate)
    assert _runs(repeated)["optimize"].status == "ran"
    assert prompts["n"] == 1
    assert gate.records[-1].scope == "session-reuse"
    other = _run(_optimize(8), tmp_path / "c", harness, gate)
    assert prompts["n"] == 2
    assert _runs(other)["optimize"].status == "ran"
    assert gate.records[-1].key == "harness:optimize"
    assert gate.records[-1].scope == "session"


def test_different_argv_fails_closed_without_a_prompter(tmp_path: Path) -> None:
    gate = ApprovalGate(decide_fn=lambda _request: ApprovalDecision.SESSION)
    harness = _Rec()
    _run(_optimize(5), tmp_path / "a", harness, gate)
    gate.interactive = False
    result = _run(_optimize(8), tmp_path / "b", harness, gate)
    assert _runs(result)["optimize"].status == "approval-denied"
    assert _runs(result)["optimize"].detail == _DENIAL
    assert [name for name, _extra in harness.calls] == ["optimize"]


def test_once_grant_is_not_reused(tmp_path: Path) -> None:
    prompts = {"n": 0}

    def decide(_request: Any) -> ApprovalDecision:
        prompts["n"] += 1
        return ApprovalDecision.ONCE

    gate = ApprovalGate(decide_fn=decide)
    harness = _Rec()
    _run(_optimize(5), tmp_path / "a", harness, gate)
    _run(_optimize(5), tmp_path / "b", harness, gate)
    assert prompts["n"] == 2
    assert {record.scope for record in gate.records} == {"once"}


def test_denial_keeps_the_operator_text(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    gate = ApprovalGate(ask_fn=lambda _question: False)
    harness = _Rec()
    result = run_plan(
        _optimize(5),
        SuperpowerContext(
            harness=harness,
            run_dir=tmp_path / "deny",
            approvals=gate,
            catalog=sample_catalog(),
            telemetry_emit=lambda kind, fields: events.append((kind, dict(fields))),
        ),
    )
    assert harness.calls == []
    assert _runs(result)["optimize"].status == "approval-denied"
    assert _runs(result)["optimize"].detail == _DENIAL
    assert ("superpower.step", {"capability": "optimize", "status": "approval-denied"}) in events


def test_idempotent_replay_does_not_ask_again(tmp_path: Path) -> None:
    prompts = {"n": 0}

    def decide(_request: Any) -> ApprovalDecision:
        prompts["n"] += 1
        return ApprovalDecision.ONCE

    gate = ApprovalGate(decide_fn=decide)
    harness = _Rec()
    ledger = JobLedger(tmp_path / "jobs")
    run_id = new_run_id()
    plan = _optimize(5)
    shared = {
        "job_ledger": ledger,
        "session_id": new_session_id(),
        "run_id": run_id,
    }
    first = _run(plan, tmp_path / "run", harness, gate, **shared)
    assert _runs(first)["optimize"].status == "ran"
    assert prompts["n"] == 1
    second = _run(plan, tmp_path / "run", harness, gate, **shared)
    assert prompts["n"] == 1
    assert len(harness.calls) == 1
    assert "idempotent replay" in _runs(second)["optimize"].detail


def test_legacy_keyword_path_binds_the_argv(tmp_path: Path) -> None:
    prompts = {"n": 0}

    def decide(_request: Any) -> ApprovalDecision:
        prompts["n"] += 1
        return ApprovalDecision.SESSION

    gate = ApprovalGate(decide_fn=decide)
    harness = _Rec()
    first = _run(_legacy_optimize(), tmp_path / "leg-a", harness, gate)
    assert _runs(first)["optimize"].status == "ran"
    assert harness.calls[0][0] == "optimize"
    assert not harness.calls[0][1]
    again = _run(_legacy_optimize(), tmp_path / "leg-b", harness, gate)
    assert _runs(again)["optimize"].status == "ran"
    assert prompts["n"] == 1
    assert gate.records[-1].scope == "session-reuse"
    typed = _run(_optimize(5), tmp_path / "typed", harness, gate)
    assert prompts["n"] == 2
    assert _runs(typed)["optimize"].status == "ran"
    assert gate.records[-1].scope == "session"
    assert gate.records[-1].key == "harness:optimize"
