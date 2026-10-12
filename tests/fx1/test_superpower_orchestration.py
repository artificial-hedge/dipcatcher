"""Catalog-gated /superpower planning, typed execution, and evidence composition."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from fx1.harness import HARNESS_REGISTRY
from fx1.honesty import validate_fx1_output
from fx1.interactive.approvals import ApprovalDecision, ApprovalGate
from fx1.interactive.budgets import RunBudget
from fx1.interactive.contracts import CapabilityCatalog
from fx1.interactive.evidence_ledger import EvidenceLedger, Verdict
from fx1.interactive.jobs import JobLedger, JobState
from fx1.interactive.runctx import new_run_id, new_session_id
from fx1.interactive.superpower import (
    CAPABILITIES,
    KIND_HARNESS,
    SuperpowerContext,
    load_catalog,
    plan_goal,
    run_plan,
)
from tests.fx1.capabilities_fixture import capability_entry, sample_catalog, sample_catalog_document

HASH = "ab" * 32


@pytest.fixture(autouse=True)
def _fx1_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))


def _ids(plan: Any) -> list[str]:
    return [step.capability_id for step in plan.typed_steps]


def _runs(result: Any) -> dict[str, Any]:
    return {run.id: run for run in result.runs}


class Rec:
    def __init__(self, stdout: str = "DATA_LABEL=SYNTHETIC\nSYNTHETIC\nok\n") -> None:
        self.calls: list[tuple[str, list[str] | None, dict[str, Any]]] = []
        self.stdout = stdout

    def run(self, name: str, extra: list[str] | None = None, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((name, extra, kwargs))
        return SimpleNamespace(
            exit_code=0,
            stdout=self.stdout,
            stderr="",
            timed_out=False,
            cancelled=False,
            truncated=False,
        )


def _ok(**overrides: Any) -> SimpleNamespace:
    payload: dict[str, Any] = {
        "exit_code": 0,
        "stdout": "DATA_LABEL=SYNTHETIC\nSYNTHETIC\nok\n",
        "stderr": "",
        "timed_out": False,
        "cancelled": False,
        "truncated": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _catalog_with(mutator: Any) -> CapabilityCatalog:
    document = sample_catalog_document()
    mutator(document)
    return CapabilityCatalog.from_json(document)


def _only_research(document: dict[str, Any]) -> None:
    entry = capability_entry(document, "research")
    entry["args"].append(
        {
            "name": "--only",
            "kind": "option",
            "type": "csv",
            "required": False,
            "repeatable": False,
            "item_type": "str",
            "help": "Family ids.",
            "choices": ["volatility", "conformal"],
        }
    )


def _ghost(document: dict[str, Any]) -> None:
    entry = json.loads(json.dumps(capability_entry(document, "vol-bench")))
    entry["id"] = "ghost-bench"
    entry["aliases"] = ["ghost bench"]
    entry["command"]["argv"] = ["ghost-bench"]
    entry["executable"] = True
    entry["executable_reason"] = ""
    document["capabilities"].append(entry)


def _family(selected: list[dict[str, Any]], exclusions: list[dict[str, str]] | None = None) -> Any:
    def selector(_goal: str) -> dict[str, Any]:
        return {
            "schema": "dipcatcher.family_surface.v1",
            "selected": selected,
            "exclusions": exclusions or [],
        }

    return selector


def _steps(payload: list[dict[str, Any]]) -> Any:
    body = json.dumps({"rationale": "model", "steps": payload})

    def model(_prompt: str) -> str:
        return body

    return model


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_catalog_plans_executable_registered_capability() -> None:
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    assert _ids(plan) == ["vol-bench"]
    assert plan.bound_catalog is not None
    assert plan.catalog_digest


def test_not_executable_is_excluded_with_catalog_reason() -> None:
    catalog = _catalog_with(
        lambda doc: capability_entry(doc, "vol-bench").update(
            {"executable": False, "executable_reason": "runner not installed"}
        )
    )
    plan = plan_goal("run vol bench on realized variance", catalog=catalog)
    assert "vol-bench" not in _ids(plan)
    assert any(
        item["capability_id"] == "vol-bench" and item["detail"] == "runner not installed"
        for item in plan.exclusions
    )
    assert "runner not installed" in plan.describe()


def test_qualified_but_unregistered_capability_is_excluded() -> None:
    plan = plan_goal("run ghost bench", catalog=_catalog_with(_ghost))
    assert "ghost-bench" not in _ids(plan)
    assert "ghost-bench" not in {command.name for command in HARNESS_REGISTRY}
    assert any("not registered in HARNESS_REGISTRY" in item["detail"] for item in plan.exclusions)
    assert "not registered in HARNESS_REGISTRY" in plan.describe()


def test_unavailable_catalog_fails_closed() -> None:
    plan = plan_goal(
        "run vol bench on realized variance",
        catalog_unavailable="capabilities list exited 2",
    )
    assert plan.catalog_unavailable
    assert not any(CAPABILITIES[cap_id]["kind"] == KIND_HARNESS for cap_id in plan.ids())
    assert "exited 2" in plan.describe()
    assert "No suitable capability" in plan.describe()


def test_unavailable_catalog_does_not_execute_harness(tmp_path: Path) -> None:
    harness = Rec()
    plan = plan_goal("run vol bench", catalog_unavailable="capabilities list exited 2")
    result = run_plan(plan, SuperpowerContext(harness=harness, run_dir=tmp_path / "run"))
    assert harness.calls == []
    assert result.honesty_violation is None


def test_no_suitable_capability_is_explained_without_web_padding() -> None:
    plan = plan_goal("summarize the history of the dutch tulip market", catalog=sample_catalog())
    assert plan.typed_steps == ()
    assert "web_research" not in plan.ids()
    assert "No suitable capability" in plan.describe()


def test_web_is_planned_only_as_heuristic_when_the_goal_asks() -> None:
    plan = plan_goal("what is the latest news on the Fed rate decision", catalog=sample_catalog())
    web = next(cap for cap in plan.capabilities if cap.id == "web_research")
    assert web.rationale.startswith("heuristic:")
    assert "heuristic" in plan.describe()


def test_keyword_planner_still_pads_web_when_no_catalog() -> None:
    plan = plan_goal("summarize the history of the dutch tulip market")
    assert "web_research" in plan.ids()
    assert plan.typed_steps == ()
    assert plan.catalog_digest == ""


def test_user_data_synthetic_only_plan_records_the_mandatory_note() -> None:
    plan = plan_goal("please run qlike scoring on my data", catalog=sample_catalog())
    assert any("SYNTHETIC" in note and "does not evaluate your data" in note for note in plan.notes)
    assert any(item["code"] == "needs_dataset" for item in plan.exclusions)
    assert "needs_dataset" in plan.describe() or any(
        item["code"] == "needs_dataset" for item in plan.exclusions
    )


def test_user_data_note_is_in_the_final_answer(tmp_path: Path) -> None:
    plan = plan_goal("please run qlike scoring on my data", catalog=sample_catalog())
    result = run_plan(
        plan,
        SuperpowerContext(harness=Rec(), run_dir=tmp_path / "run", catalog=sample_catalog()),
    )
    assert "does not evaluate your data" in result.summary
    assert "no qualified dataset-bound runner matched" in result.summary
    assert "SYNTHETIC" in result.summary


def test_refinement_rejects_unknown_id() -> None:
    plan = plan_goal(
        "run vol bench on realized variance",
        catalog=sample_catalog(),
        model_fn=_steps([{"capability": "no-such-cap", "args": {}, "why": "nope"}]),
    )
    assert "no-such-cap" not in plan.ids()
    assert any(item["code"] == "unknown_id" for item in plan.exclusions)


def test_refinement_rejects_invalid_arg() -> None:
    plan = plan_goal(
        "run vol bench on realized variance",
        catalog=sample_catalog(),
        model_fn=_steps([{"capability": "vol-bench", "args": {"--not-a-flag": 1}, "why": "bad"}]),
    )
    assert "vol-bench" not in _ids(plan)
    assert any(item["code"] == "invalid_args" for item in plan.exclusions)


def test_refinement_rejects_over_cap_arg() -> None:
    plan = plan_goal(
        "run vol bench on realized variance",
        catalog=sample_catalog(),
        model_fn=_steps([{"capability": "vol-bench", "args": {"--n-origins": 100}, "why": "big"}]),
    )
    assert "vol-bench" not in _ids(plan)
    assert any("out_of_range" in item["detail"] for item in plan.exclusions)


def test_refinement_rejects_shell_string() -> None:
    plan = plan_goal(
        "run vol bench on realized variance",
        catalog=sample_catalog(),
        model_fn=_steps(
            [{"capability": "vol-bench", "args": {"--models": "har;rm"}, "why": "bad"}]
        ),
    )
    assert "vol-bench" not in _ids(plan)
    assert any(
        "shell" in item["detail"] or item["code"] == "invalid_args" for item in plan.exclusions
    )


def test_malformed_catalog_refinement_keeps_deterministic_selection() -> None:
    plan = plan_goal(
        "run vol bench on realized variance",
        catalog=sample_catalog(),
        model_fn=lambda _prompt: "not json",
    )
    assert _ids(plan) == ["vol-bench"]
    assert plan.source == "deterministic"


def test_empty_family_selection_excludes_research() -> None:
    plan = plan_goal(
        "run the full scorecard",
        catalog=sample_catalog(),
        family_selector=_family([], [{"id": "volatility", "reason": "not executable"}]),
    )
    assert "research" not in _ids(plan)
    assert plan.research_constrained is True
    assert plan.families == []
    assert "volatility" in plan.describe()


def test_family_surface_exception_excludes_research() -> None:
    def boom(_goal: str) -> dict[str, Any]:
        raise RuntimeError("family surface down")

    plan = plan_goal("run the full scorecard", catalog=sample_catalog(), family_selector=boom)
    assert "research" not in _ids(plan)
    assert any("family surface refused" in item["detail"] for item in plan.exclusions)


def test_families_without_only_argument_do_not_run_the_full_scorecard() -> None:
    plan = plan_goal(
        "run the full scorecard",
        catalog=sample_catalog(),
        family_selector=_family(
            [{"id": "volatility", "executable": True, "qualification": "required"}]
        ),
    )
    assert "research" not in _ids(plan)
    assert any("no --only" in item["detail"] for item in plan.exclusions)


def test_families_bind_typed_only_argument() -> None:
    plan = plan_goal(
        "run the full scorecard",
        catalog=_catalog_with(_only_research),
        family_selector=_family(
            [{"id": "volatility", "executable": True, "qualification": "required"}]
        ),
    )
    research = next(step for step in plan.typed_steps if step.capability_id == "research")
    assert research.args["--only"] == ["volatility"]
    assert plan.families == ["volatility"]


def test_family_choice_outside_schema_is_excluded() -> None:
    plan = plan_goal(
        "run the full scorecard",
        catalog=_catalog_with(_only_research),
        family_selector=_family(
            [{"id": "not_allowed", "executable": True, "qualification": "required"}]
        ),
    )
    assert "research" not in _ids(plan)
    assert any("only" in item["detail"] for item in plan.exclusions)


def test_load_catalog_success_and_redacted_failure(tmp_path: Path) -> None:
    document = json.dumps(sample_catalog_document())

    class Ok:
        def run(self, name: str, args: list[str]) -> SimpleNamespace:
            assert (name, args) == ("capabilities", ["list", "--json"])
            return SimpleNamespace(exit_code=0, stdout=document)

    loaded = load_catalog(Ok(), config_dir=tmp_path / "cfg")
    assert loaded.catalog is not None
    assert loaded.reason is None
    assert loaded.catalog.get("vol-bench") is not None

    class Boom:
        def run(self, name: str, args: list[str]) -> SimpleNamespace:
            raise RuntimeError("bearer sk-abcdefghijklmnopqrstuvwxyz1234")

    failed = load_catalog(Boom(), config_dir=tmp_path / "cfg2")
    assert failed.catalog is None
    assert failed.reason is not None
    assert "sk-" not in failed.reason
    assert "REDACTED" in failed.reason


def test_load_catalog_nonzero_exit_fails_closed(tmp_path: Path) -> None:
    class Bad:
        def run(self, name: str, args: list[str]) -> SimpleNamespace:
            return SimpleNamespace(exit_code=2, stdout="nope")

    failed = load_catalog(Bad(), config_dir=tmp_path / "cfg")
    assert failed.catalog is None
    assert failed.reason is not None
    assert "2" in failed.reason


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_typed_argv_is_a_list_and_forces_out_dir_inside_run_dir(tmp_path: Path) -> None:
    harness = Rec()
    run_dir = tmp_path / "run"
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    run_plan(
        plan,
        SuperpowerContext(harness=harness, run_dir=run_dir, catalog=sample_catalog()),
    )
    assert len(harness.calls) == 1
    name, extra, kwargs = harness.calls[0]
    assert name == "vol-bench"
    assert isinstance(extra, list)
    assert "--out-dir" in extra
    out = Path(extra[extra.index("--out-dir") + 1])
    assert run_dir in out.parents
    assert "receipts" not in extra
    assert all(";" not in token and "|" not in token for token in extra)
    assert kwargs["env_extra"]["DIPCATCHER_RUN_ID"]
    assert kwargs["step"]


def test_dependency_failure_skips_dependents(tmp_path: Path) -> None:
    class DoctorFails(Rec):
        def run(self, name: str, extra: list[str] | None = None, **kwargs: Any) -> SimpleNamespace:
            self.calls.append((name, extra, kwargs))
            if name == "doctor":
                return _ok(exit_code=2, stdout="", stderr="boom")
            return _ok()

    harness = DoctorFails()
    plan = plan_goal("run the full scorecard", catalog=sample_catalog())
    result = run_plan(plan, SuperpowerContext(harness=harness, run_dir=tmp_path / "run"))
    assert [call[0] for call in harness.calls] == ["doctor"]
    assert _runs(result)["research"].status == "skipped"
    assert "prerequisite doctor failed" in _runs(result)["research"].detail


def test_cancel_mid_run_skips_dependents_and_confirms_the_job(tmp_path: Path) -> None:
    class CancelOnDoctor(Rec):
        def run(self, name: str, extra: list[str] | None = None, **kwargs: Any) -> SimpleNamespace:
            self.calls.append((name, extra, kwargs))
            event = kwargs.get("cancel_event")
            if event is not None:
                event.set()
            return _ok(exit_code=130, cancelled=True, stderr="cancelled")

    harness = CancelOnDoctor()
    ledger = JobLedger(tmp_path / "jobs")
    plan = plan_goal("run the full scorecard", catalog=sample_catalog())
    ctx = SuperpowerContext(
        harness=harness,
        run_dir=tmp_path / "run",
        job_ledger=ledger,
        session_id=new_session_id(),
        run_id=new_run_id(),
        cancel_event=threading.Event(),
    )
    result = run_plan(plan, ctx)
    assert [call[0] for call in harness.calls] == ["doctor"]
    assert _runs(result)["doctor"].status == "cancelled"
    assert _runs(result)["research"].status == "skipped"
    job = ledger.get(result.job_id or "")
    assert job.state is JobState.CANCELLED
    assert job.state is not JobState.CANCELLING


def test_timeout_and_truncation_are_explicit(tmp_path: Path) -> None:
    class Slow(Rec):
        def run(self, name: str, extra: list[str] | None = None, **kwargs: Any) -> SimpleNamespace:
            self.calls.append((name, extra, kwargs))
            return _ok(exit_code=124, timed_out=True, truncated=True, stderr="deadline")

    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    result = run_plan(plan, SuperpowerContext(harness=Slow(), run_dir=tmp_path / "run"))
    run = _runs(result)["vol-bench"]
    assert run.status == "timed-out"
    assert "truncated" in run.detail
    assert "timed-out" in run.detail


def test_nonzero_exit_is_not_success_even_if_stdout_says_verified(tmp_path: Path) -> None:
    class Failed(Rec):
        def run(self, name: str, extra: list[str] | None = None, **kwargs: Any) -> SimpleNamespace:
            self.calls.append((name, extra, kwargs))
            return _ok(exit_code=2, stdout="verified receipt\n", stderr="nope")

    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    result = run_plan(plan, SuperpowerContext(harness=Failed(), run_dir=tmp_path / "run"))
    assert _runs(result)["vol-bench"].status == "failed"
    assert result.status != "verified"


def test_consequential_deny_once_session_and_no_prompter(tmp_path: Path) -> None:
    plan = plan_goal("run a hyperparameter search", catalog=sample_catalog())
    assert "optimize" in _ids(plan)

    denied = Rec()
    denied_result = run_plan(
        plan,
        SuperpowerContext(
            harness=denied,
            run_dir=tmp_path / "deny",
            approvals=ApprovalGate(ask_fn=lambda _question: False),
        ),
    )
    assert denied.calls == []
    assert _runs(denied_result)["optimize"].status == "approval-denied"

    calls = {"n": 0}

    def once(request: Any) -> ApprovalDecision:
        calls["n"] += 1
        return ApprovalDecision.ONCE

    gate = ApprovalGate(decide_fn=once)
    run_plan(plan, SuperpowerContext(harness=Rec(), run_dir=tmp_path / "once-a", approvals=gate))
    run_plan(plan, SuperpowerContext(harness=Rec(), run_dir=tmp_path / "once-b", approvals=gate))
    assert calls["n"] == 2
    assert {record.scope for record in gate.records} == {"once"}

    session_calls = {"n": 0}

    def session(request: Any) -> ApprovalDecision:
        session_calls["n"] += 1
        return ApprovalDecision.SESSION

    session_gate = ApprovalGate(decide_fn=session)
    run_plan(
        plan,
        SuperpowerContext(harness=Rec(), run_dir=tmp_path / "sess-a", approvals=session_gate),
    )
    run_plan(
        plan,
        SuperpowerContext(harness=Rec(), run_dir=tmp_path / "sess-b", approvals=session_gate),
    )
    assert session_calls["n"] == 1
    assert any(record.scope == "session-reuse" for record in session_gate.records)
    assert any(record.key == "harness:optimize" for record in session_gate.records)


def test_no_prompter_records_deny(tmp_path: Path) -> None:
    gate = ApprovalGate()
    plan = plan_goal("run a hyperparameter search", catalog=sample_catalog())
    result = run_plan(
        plan, SuperpowerContext(harness=Rec(), run_dir=tmp_path / "run", approvals=gate)
    )
    assert _runs(result)["optimize"].status == "approval-denied"
    assert gate.records[0].scope == "no-prompter"
    assert gate.records[0].key == "harness:optimize"


def test_shell_string_is_rejected_at_execution(tmp_path: Path) -> None:
    harness = Rec()
    from dataclasses import replace

    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    step = plan.typed_steps[0]
    plan.typed_steps = (replace(step, args=MappingProxyType({"--models": "har;rm"})),)
    result = run_plan(plan, SuperpowerContext(harness=harness, run_dir=tmp_path / "run"))
    assert harness.calls == []
    assert _runs(result)["vol-bench"].status == "failed"
    assert (
        "shell" in _runs(result)["vol-bench"].detail
        or "bad_type" in _runs(result)["vol-bench"].detail
    )


def test_job_ledger_succeeds_and_replays_idempotent_steps(tmp_path: Path) -> None:
    harness = Rec()
    ledger = JobLedger(tmp_path / "jobs")
    run_id = new_run_id()
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    ctx = SuperpowerContext(
        harness=harness,
        run_dir=tmp_path / "run",
        job_ledger=ledger,
        session_id=new_session_id(),
        run_id=run_id,
    )
    first = run_plan(plan, ctx)
    job = ledger.get(first.job_id or "")
    assert job.state is JobState.SUCCEEDED
    assert job.kind == "superpower"
    assert job.steps[0].state is JobState.SUCCEEDED
    assert job.steps[0].idempotency_key == f"{run_id}:{plan.typed_steps[0].step_key}"
    second = run_plan(
        plan,
        SuperpowerContext(
            harness=harness,
            run_dir=tmp_path / "run",
            job_ledger=ledger,
            session_id=ctx.session_id,
            run_id=run_id,
        ),
    )
    assert len(harness.calls) == 1
    assert "idempotent replay" in _runs(second)["vol-bench"].detail


def test_retries_are_bounded_by_the_budget(tmp_path: Path) -> None:
    class Flaky(Rec):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def run(self, name: str, extra: list[str] | None = None, **kwargs: Any) -> SimpleNamespace:
            self.calls.append((name, extra, kwargs))
            self.n += 1
            if self.n == 1:
                return _ok(exit_code=2, stdout="", stderr="transient")
            return _ok()

    flaky = Flaky()
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    result = run_plan(
        plan,
        SuperpowerContext(
            harness=flaky,
            run_dir=tmp_path / "run",
            budget=RunBudget(max_retries=1, max_steps=8),
        ),
    )
    assert flaky.n == 2
    assert _runs(result)["vol-bench"].status == "ran"

    once = Flaky()
    stopped = run_plan(
        plan,
        SuperpowerContext(
            harness=once,
            run_dir=tmp_path / "once",
            budget=RunBudget(max_retries=0, max_steps=8),
        ),
    )
    assert once.n == 1
    assert _runs(stopped)["vol-bench"].status == "failed"


def test_family_budget_skips_research(tmp_path: Path) -> None:
    harness = Rec()
    plan = plan_goal(
        "run the full scorecard",
        catalog=_catalog_with(_only_research),
        family_selector=_family(
            [{"id": "volatility", "executable": True, "qualification": "required"}]
        ),
    )
    result = run_plan(
        plan,
        SuperpowerContext(
            harness=harness,
            run_dir=tmp_path / "run",
            budget=RunBudget(max_families=0, max_steps=8),
        ),
    )
    assert "research" not in [call[0] for call in harness.calls]
    assert _runs(result)["research"].status == "skipped"
    assert "family budget" in _runs(result)["research"].detail


def test_owned_runner_spawn_attaches_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class FakeOwned:
        def __init__(self, *, emit: Any = None, registry: Any = None) -> None:
            self.emit = emit
            self.supports_extended = True

        def __call__(self, argv: list[str], timeout_s: int, **kwargs: Any) -> SimpleNamespace:
            captured["step"] = kwargs.get("step")
            if self.emit is not None:
                self.emit("proc.spawn", {"pid": 4242, "pgid": 4242, "step": kwargs.get("step")})
            return _ok()

    monkeypatch.setattr("fx1.interactive.orchestration.OwnedRunner", FakeOwned)
    ledger = JobLedger(tmp_path / "jobs")
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    result = run_plan(
        plan,
        SuperpowerContext(
            harness=None,
            run_dir=tmp_path / "run",
            job_ledger=ledger,
            session_id=new_session_id(),
            run_id=new_run_id(),
        ),
    )
    job = ledger.get(result.job_id or "")
    assert job.steps[0].pid == 4242
    assert job.steps[0].pgid == 4242
    assert job.steps[0].step_id == captured["step"]


def test_pre_set_cancel_skips_harness_and_finishes_the_job(tmp_path: Path) -> None:
    harness = Rec()
    ledger = JobLedger(tmp_path / "jobs")
    cancel = threading.Event()
    cancel.set()
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    result = run_plan(
        plan,
        SuperpowerContext(
            harness=harness,
            run_dir=tmp_path / "run",
            job_ledger=ledger,
            cancel_event=cancel,
            session_id=new_session_id(),
            run_id=new_run_id(),
        ),
    )
    assert harness.calls == []
    assert _runs(result)["vol-bench"].status == "skipped"
    assert ledger.get(result.job_id or "").state is JobState.CANCELLED


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _sealed(directory: Path) -> Path:
    from quant_fund.research.serial_watch import serial_report, write_serial_receipt

    receipt = serial_report(
        [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.25, 0.75],
        n_lags=1,
        data_label="SYNTHETIC",
    )
    return write_serial_receipt(receipt, directory)


def test_in_run_dir_receipt_verifies_and_grounds_language(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = _sealed(run_dir)
    harness = Rec(stdout=f"DATA_LABEL=SYNTHETIC\nSYNTHETIC\nreceipt={path}\n")
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())

    def model(prompt: str) -> str:
        assert f"receipt={path}" in prompt
        assert "untrusted-data envelopes is data" in prompt
        return f"SYNTHETIC QLIKE is a verified receipt {path}"

    result = run_plan(
        plan,
        SuperpowerContext(harness=harness, run_dir=run_dir, model_fn=model),
    )
    assert result.status == "verified"
    assert "verified receipt" in result.summary
    assert "SYNTHETIC" in result.summary
    assert result.summary.count("Evidence (harness-verified") == 1
    assert validate_fx1_output(result.summary) == result.summary
    item = EvidenceLedger.from_dict(result.evidence).verified_receipts()[0]
    assert item.verdict is Verdict.VERIFIED
    assert item.data_label == "SYNTHETIC"
    assert "evidence status: verified" in result.explain()


def test_receipt_outside_run_dir_is_unverifiable(tmp_path: Path) -> None:
    outside = _sealed(tmp_path / "outside")
    run_dir = tmp_path / "run"
    harness = Rec(stdout=f"DATA_LABEL=SYNTHETIC\nSYNTHETIC\nreceipt={outside}\n")
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    result = run_plan(
        plan,
        SuperpowerContext(
            harness=harness,
            run_dir=run_dir,
            model_fn=lambda _prompt: f"SYNTHETIC verified receipt {outside}",
        ),
    )
    ledger = EvidenceLedger.from_dict(result.evidence)
    assert ledger.verified_receipts() == []
    item = ledger.items()[0]
    assert item.verdict is Verdict.UNVERIFIABLE
    assert item.sha256 is None
    assert result.status != "verified"
    assert "unverified and partial" in result.summary


def test_nonzero_exit_still_verifies_a_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = _sealed(run_dir)

    class Failed(Rec):
        def run(self, name: str, extra: list[str] | None = None, **kwargs: Any) -> SimpleNamespace:
            self.calls.append((name, extra, kwargs))
            return _ok(exit_code=3, stdout=f"DATA_LABEL=SYNTHETIC\nSYNTHETIC\nreceipt={path}\n")

    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    result = run_plan(
        plan,
        SuperpowerContext(
            harness=Failed(),
            run_dir=run_dir,
            model_fn=lambda _prompt: f"SYNTHETIC command failed; verified receipt {path}",
        ),
    )
    assert _runs(result)["vol-bench"].status == "failed"
    assert EvidenceLedger.from_dict(result.evidence).verified_receipts()
    assert result.status == "verified"


def test_partial_family_record_is_never_verified(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    partial = run_dir / "family.json"
    partial.write_text('{"verified": true}', encoding="utf-8")
    harness = Rec(stdout=f"DATA_LABEL=SYNTHETIC\nSYNTHETIC\njson={partial}\nverified=false\n")
    plan = plan_goal("run the full scorecard", catalog=sample_catalog())
    # doctor would also run; point the research stdout by failing closed on doctor?
    # Use a harness that emits the partial record for every command.
    result = run_plan(
        plan,
        SuperpowerContext(
            harness=harness,
            run_dir=run_dir,
            model_fn=lambda _prompt: "SYNTHETIC partial family record verified=false",
        ),
    )
    ledger = EvidenceLedger.from_dict(result.evidence)
    artifacts = [item for item in ledger.items() if "partial family record" in item.detail]
    assert artifacts
    assert all(item.verdict is not Verdict.VERIFIED for item in artifacts)
    assert all("verified=false" in item.detail for item in artifacts)
    assert ledger.has_verified() is False


def test_unknown_hash_is_flagged_and_forbidden_headline_is_withheld(tmp_path: Path) -> None:
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    cited = run_plan(
        plan,
        SuperpowerContext(
            harness=Rec(),
            run_dir=tmp_path / "run",
            model_fn=lambda _prompt: f"SYNTHETIC note {HASH}",
        ),
    )
    assert "unknown_hash" in cited.summary
    assert cited.status != "verified"
    assert cited.summary.count("Evidence (harness-verified") == 1

    withheld = run_plan(
        plan,
        SuperpowerContext(
            harness=Rec(),
            run_dir=tmp_path / "withheld",
            model_fn=lambda _prompt: "The sharpe ratio is 2.1",
        ),
    )
    assert withheld.status == "withheld"
    assert withheld.honesty_violation is not None
    assert "forbidden" in withheld.honesty_violation
    assert "2.1" not in withheld.summary
    assert "sharpe ratio is" not in withheld.summary.lower()
    assert withheld.summary.count("Evidence (harness-verified") == 1
    assert validate_fx1_output(withheld.summary) == withheld.summary


def test_verified_language_without_a_receipt_is_downgraded(tmp_path: Path) -> None:
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    result = run_plan(
        plan,
        SuperpowerContext(
            harness=Rec(),
            run_dir=tmp_path / "run",
            model_fn=lambda _prompt: "SYNTHETIC this is a verified receipt",
        ),
    )
    assert result.status != "verified"
    assert "unverified and partial" in result.summary
    assert "evidence status:" in result.explain()


def test_offline_summary_is_grounded_once(tmp_path: Path) -> None:
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    result = run_plan(plan, SuperpowerContext(harness=Rec(), run_dir=tmp_path / "run"))
    assert "no model endpoint" in result.summary
    assert result.summary.count("Evidence (harness-verified") == 1
    assert "SYNTHETIC" in result.summary
    assert validate_fx1_output(result.summary) == result.summary
    lowered = result.summary.lower()
    for token in ("sharpe", "sortino", "calmar", "p&l", " nav", "pnl"):
        assert token not in lowered


def test_run_dir_is_private_on_posix(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    plan = plan_goal("run vol bench on realized variance", catalog=sample_catalog())
    run_plan(plan, SuperpowerContext(harness=Rec(), run_dir=run_dir))
    if os.name == "posix":
        assert os.stat(run_dir).st_mode & 0o777 == 0o700


# ---------------------------------------------------------------------------
# Capability id differs from ``command.argv``; absolute confined paths
# ---------------------------------------------------------------------------


def _selective(document: dict[str, Any]) -> None:
    """Add ``research-only`` (argv ``["research"]``) beside the full ``research``."""
    entry = json.loads(json.dumps(capability_entry(document, "research")))
    entry["id"] = "research-only"
    entry["aliases"] = ["research only", "selective families"]
    entry["tasks"] = ["selective_family_run"]
    entry["dependencies"] = []
    entry["args"].append(
        {
            "name": "--only",
            "kind": "option",
            "type": "csv",
            "required": True,
            "repeatable": False,
            "item_type": "str",
            "help": "Family ids.",
            "choices": ["volatility", "conformal"],
        }
    )
    document["capabilities"].append(entry)


def _selector_for(*ids: str) -> Any:
    return _family([{"id": item, "executable": True, "qualification": "required"} for item in ids])


def test_selective_capability_binds_only_and_runs_argv_zero(tmp_path: Path) -> None:
    catalog = _catalog_with(_selective)
    plan = plan_goal(
        "run research only selective families",
        catalog=catalog,
        family_selector=_selector_for("conformal"),
    )
    assert _ids(plan) == ["research-only"]
    assert plan.families == ["conformal"]
    harness = Rec(stdout="DATA_LABEL=SYNTHETIC\nSYNTHETIC\njson=/nowhere/x.json\n")
    ledger = JobLedger(tmp_path / "jobs")
    result = run_plan(
        plan,
        SuperpowerContext(
            harness=harness,
            catalog=catalog,
            run_dir=tmp_path / "run",
            job_ledger=ledger,
            session_id=new_session_id(),
            run_id=new_run_id(),
        ),
    )
    name, extra, _kwargs = harness.calls[0]
    assert name == "research"
    assert extra is not None and "research-only" not in extra
    assert extra[extra.index("--only") + 1] == "conformal"
    job = ledger.get(result.job_id or "")
    assert job.steps[0].name == "research-only"
    assert job.steps[0].argv_summary[0] == "research"
    assert _runs(result)["research-only"].status == "ran"
    assert result.status != "verified"


def test_full_research_is_not_widened_when_selective_sibling_exists() -> None:
    plan = plan_goal(
        "run the full scorecard and research only selective families",
        catalog=_catalog_with(_selective),
        family_selector=_selector_for("conformal"),
    )
    assert "research" not in _ids(plan)
    assert plan.families == ["conformal"] or plan.families == []


def test_registry_gate_uses_argv_zero_not_the_capability_id() -> None:
    def alias(document: dict[str, Any]) -> None:
        entry = json.loads(json.dumps(capability_entry(document, "vol-bench")))
        entry.update({"id": "vol-alias", "aliases": ["vol alias bench"]})
        document["capabilities"].append(entry)

    def ghost_argv(document: dict[str, Any]) -> None:
        entry = capability_entry(document, "vol-bench")
        entry["command"]["argv"] = ["ghost-bench"]

    aliased = plan_goal("run vol alias bench", catalog=_catalog_with(alias))
    assert "vol-alias" in _ids(aliased)
    ghost = plan_goal("run vol bench on realized variance", catalog=_catalog_with(ghost_argv))
    assert "vol-bench" not in _ids(ghost)
    assert any("not registered in HARNESS_REGISTRY" in item["detail"] for item in ghost.exclusions)


def test_confined_absolute_out_dir_survives_argv_rebuild(tmp_path: Path) -> None:
    def out_dir_arg(document: dict[str, Any]) -> None:
        entry = capability_entry(document, "vol-bench")
        entry["allowed_args"]["--out-dir"] = {}
        entry["args"].append(
            {
                "name": "--out-dir",
                "kind": "option",
                "type": "path",
                "required": True,
                "repeatable": False,
                "path_role": "output_dir",
                "help": "Receipt directory.",
            }
        )

    run_dir = tmp_path / "run"
    catalog = _catalog_with(out_dir_arg)
    roots = {"output_dir": [run_dir]}
    plan = plan_goal("run vol bench on realized variance", catalog=catalog, roots=roots)
    harness = Rec()
    result = run_plan(
        plan,
        SuperpowerContext(harness=harness, catalog=catalog, run_dir=run_dir, roots=roots),
    )
    assert _runs(result)["vol-bench"].status == "ran"
    extra = harness.calls[0][1]
    assert extra is not None and extra.count("--out-dir") == 1
    assert Path(extra[extra.index("--out-dir") + 1]).is_absolute()
    assert run_dir in Path(extra[extra.index("--out-dir") + 1]).parents
