"""Durable job ledger: transitions, restart reconciliation, cancel, idempotency."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fx1.interactive import jobs as jobs_mod
from fx1.interactive.atomicstore import StoreCorruptError
from fx1.interactive.jobs import (
    SCHEMA_VERSION,
    TERMINAL_STATES,
    IllegalTransition,
    JobLedger,
    JobNotFound,
    JobState,
    ProcessNote,
    ReconcileReport,
    SystemProcessProbe,
    terminate_process,
)
from fx1.interactive.runctx import new_job_id, new_run_id, new_session_id

SECRET = "sk-" + "a" * 20


class MutableClock:
    def __init__(self) -> None:
        self.moment = datetime(2026, 10, 12, 3, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.moment


class FakeProbe:
    def __init__(self) -> None:
        self.alive: dict[int, bool] = {}
        self.tokens: dict[int, str | None] = {}
        self.fail = False

    def is_alive(self, pid: int) -> bool:
        if self.fail:
            raise OSError("probe down")
        return self.alive.get(pid, False)

    def token(self, pid: int) -> str | None:
        if self.fail:
            raise OSError("probe down")
        return self.tokens.get(pid)


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "home"))


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def ledger(tmp_path: Path, clock: MutableClock) -> JobLedger:
    return JobLedger(tmp_path / "jobs", clock=clock)


def _reach(ledger: JobLedger, job_id: str, state: JobState) -> None:
    if state == JobState.QUEUED:
        return
    if state == JobState.CANCELLED:
        ledger.transition(job_id, JobState.CANCELLED)
        return
    ledger.transition(job_id, JobState.RUNNING)
    if state != JobState.RUNNING:
        ledger.transition(job_id, state)


LEGAL = [
    (JobState.QUEUED, JobState.RUNNING),
    (JobState.QUEUED, JobState.CANCELLED),
    (JobState.RUNNING, JobState.CANCELLING),
    (JobState.RUNNING, JobState.SUCCEEDED),
    (JobState.RUNNING, JobState.FAILED),
    (JobState.RUNNING, JobState.TIMED_OUT),
    (JobState.RUNNING, JobState.INTERRUPTED),
    (JobState.CANCELLING, JobState.CANCELLED),
    (JobState.CANCELLING, JobState.FAILED),
    (JobState.CANCELLING, JobState.INTERRUPTED),
]

ILLEGAL = [
    (JobState.QUEUED, JobState.SUCCEEDED),
    (JobState.QUEUED, JobState.FAILED),
    (JobState.QUEUED, JobState.CANCELLING),
    (JobState.QUEUED, JobState.INTERRUPTED),
    (JobState.QUEUED, JobState.TIMED_OUT),
    (JobState.RUNNING, JobState.QUEUED),
    (JobState.RUNNING, JobState.CANCELLED),
    (JobState.CANCELLING, JobState.RUNNING),
    (JobState.CANCELLING, JobState.SUCCEEDED),
    (JobState.CANCELLING, JobState.TIMED_OUT),
    (JobState.SUCCEEDED, JobState.FAILED),
    (JobState.FAILED, JobState.RUNNING),
    (JobState.CANCELLED, JobState.QUEUED),
    (JobState.INTERRUPTED, JobState.RUNNING),
    (JobState.TIMED_OUT, JobState.RUNNING),
]


def test_create_roundtrip_schema_and_private_files(
    ledger: JobLedger, tmp_path: Path, clock: MutableClock
) -> None:
    session_id = new_session_id(now=clock.moment)
    run_id = new_run_id(now=clock.moment)
    job = ledger.create(
        kind="superpower",
        goal="bench vol",
        session_id=session_id,
        run_id=run_id,
        budget={"max_steps": 3},
    )
    assert job.state == JobState.QUEUED
    assert job.version == str(SCHEMA_VERSION)
    assert job.revision == 1
    assert job.session_id == session_id and job.run_id == run_id
    path = tmp_path / "jobs" / f"{job.job_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["goal"] == "bench vol"
    assert ledger.get(job.job_id) == job
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    caller_copy = ledger.get(job.job_id)
    caller_copy.goal = "mutated"
    assert ledger.get(job.job_id).goal == "bench vol"


def test_default_root_follows_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: MutableClock
) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    job = JobLedger(clock=clock).create(kind="turn", goal="hello")
    assert (tmp_path / "cfg" / "jobs" / f"{job.job_id}.json").is_file()


def test_goals_argv_errors_and_receipts_are_redacted(ledger: JobLedger, tmp_path: Path) -> None:
    job = ledger.create(
        kind="turn",
        goal=f"use {SECRET} and api_key=supersecretvalue",
        budget={"max_steps": 1, "api_key": "abcdefghij"},
    )
    ledger.transition(job.job_id, JobState.RUNNING)
    step = ledger.begin_step(
        job.job_id,
        name="vol-bench",
        argv=["tool", "--api-key", SECRET, f"--token={SECRET}"],
        idempotent=False,
    )
    done = ledger.finish_step(
        job.job_id,
        step.step_id,
        state=JobState.FAILED,
        exit_code=1,
        receipt_paths=[f"https://user:{SECRET}@example.com/r"],
        detail=f"bearer {SECRET}",
    )
    failed = ledger.transition(job.job_id, JobState.FAILED, error=f"boom {SECRET}")
    blob = (tmp_path / "jobs" / f"{job.job_id}.json").read_text(encoding="utf-8")
    assert SECRET not in blob
    assert "supersecretvalue" not in blob
    assert "abcdefghij" not in blob
    assert SECRET not in repr(failed)
    assert done.argv_summary[2] == "[REDACTED]"
    assert SECRET not in done.detail
    assert SECRET not in (failed.error or "")
    assert failed.budget is not None
    assert failed.budget["api_key"] == "[REDACTED]"
    assert job.job_id in blob


def test_validation_and_missing_job(ledger: JobLedger, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ledger.create(kind="Nope", goal="g")
    with pytest.raises(ValueError):
        ledger.create(kind="turn", goal="g", session_id=new_job_id())
    with pytest.raises(ValueError):
        ledger.create(kind="turn", goal="g", run_id=new_session_id())
    with pytest.raises(TypeError):
        ledger.create(kind="turn", goal=12)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ledger.create(kind="turn", goal="g", budget={"when": object()})
    job = ledger.create(kind="exec", goal="g")
    with pytest.raises(JobNotFound) as missing:
        ledger.get(new_job_id())
    assert missing.value.job_id
    with pytest.raises(ValueError):
        ledger.get("../" + job.job_id)
    with pytest.raises(ValueError):
        ledger.get("j-20261012T024500Z-zzzzzzzz")
    assert not (tmp_path / "etc").exists()
    with pytest.raises(ValueError):
        ledger.begin_step(job.job_id, name="bad name", argv=["x"], idempotent=False)
    with pytest.raises(ValueError):
        ledger.begin_step(
            job.job_id, name="vol-bench", argv=["x"], idempotent=False, step_index=1000
        )
    assert ledger.get(job.job_id).state == JobState.QUEUED
    assert ledger.get(job.job_id).steps == []


@pytest.mark.parametrize(("source", "target"), LEGAL)
def test_legal_transitions(ledger: JobLedger, source: JobState, target: JobState) -> None:
    job = ledger.create(kind="turn", goal="g")
    _reach(ledger, job.job_id, source)
    before = ledger.get(job.job_id)
    updated = ledger.transition(job.job_id, target, error="context")
    assert updated.state == target
    assert updated.revision == before.revision + 1
    if target == JobState.FAILED:
        assert updated.error == "context"
    assert ledger.get(job.job_id).state == target


@pytest.mark.parametrize(("source", "target"), ILLEGAL)
def test_illegal_transitions_do_not_persist(
    ledger: JobLedger, source: JobState, target: JobState
) -> None:
    job = ledger.create(kind="turn", goal="g")
    _reach(ledger, job.job_id, source)
    before = ledger.get(job.job_id)
    with pytest.raises(IllegalTransition) as info:
        ledger.transition(job.job_id, target)
    assert info.value.current == source
    assert info.value.target == target
    after = ledger.get(job.job_id)
    assert after.state == before.state
    assert after.revision == before.revision
    assert after.updated_at == before.updated_at


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES, key=lambda item: item.value))
def test_terminal_reassertion_is_a_noop(
    ledger: JobLedger, clock: MutableClock, state: JobState
) -> None:
    job = ledger.create(kind="turn", goal="g")
    _reach(ledger, job.job_id, state)
    before = ledger.get(job.job_id)
    clock.moment += timedelta(days=1)
    again = ledger.transition(job.job_id, state, error="ignored")
    assert again.state == state
    assert again.revision == before.revision
    assert again.updated_at == before.updated_at
    assert again.error == before.error


def test_same_nonterminal_state_is_a_noop(ledger: JobLedger, clock: MutableClock) -> None:
    job = ledger.create(kind="turn", goal="g")
    running = ledger.transition(job.job_id, JobState.RUNNING)
    clock.moment += timedelta(hours=2)
    again = ledger.transition(job.job_id, JobState.RUNNING, error="nope")
    assert again.revision == running.revision
    assert again.updated_at == running.updated_at
    assert again.error is None


def test_cancel_is_kept_until_confirmed(ledger: JobLedger, clock: MutableClock) -> None:
    probe = FakeProbe()
    probe.alive[41] = True
    probe.tokens[41] = "tok"
    owned = JobLedger(ledger.root, clock=clock, probe=probe)
    job = owned.create(kind="research", goal="look up")
    step = owned.begin_step(job.job_id, name="web-research", argv=["fetch"], idempotent=False)
    owned.attach_process(job.job_id, step.step_id, pid=41, pgid=41)
    first = owned.request_cancel(job.job_id)
    assert first.state == JobState.CANCELLING
    assert first.cancel_requested_at
    clock.moment += timedelta(hours=3)
    second = owned.request_cancel(job.job_id)
    assert second.cancel_requested_at == first.cancel_requested_at
    assert second.revision == first.revision
    with pytest.raises(IllegalTransition, match="still running"):
        owned.confirm_terminated(job.job_id)
    assert owned.get(job.job_id).state == JobState.CANCELLING
    probe.tokens[41] = None
    with pytest.raises(IllegalTransition, match="unverifiable"):
        owned.confirm_terminated(job.job_id)
    probe.tokens[41] = "recycled"
    confirmed = owned.confirm_terminated(job.job_id)
    assert confirmed.state == JobState.CANCELLED
    assert confirmed.cancel_requested_at == first.cancel_requested_at
    assert confirmed.steps[0].state == JobState.CANCELLED
    clock.moment += timedelta(minutes=1)
    repeated = owned.confirm_terminated(job.job_id)
    assert repeated.revision == confirmed.revision


def test_confirm_when_process_is_dead_and_queued_cancel_completes(
    ledger: JobLedger, clock: MutableClock
) -> None:
    probe = FakeProbe()
    probe.alive[11] = False
    probe.tokens[11] = "tok"
    owned = JobLedger(ledger.root, clock=clock, probe=probe)
    queued = owned.create(kind="turn", goal="never started")
    cancelled = owned.request_cancel(queued.job_id)
    assert cancelled.state == JobState.CANCELLED
    assert cancelled.cancel_requested_at
    with pytest.raises(IllegalTransition):
        owned.request_cancel(queued.job_id)
    job = owned.create(kind="superpower", goal="run")
    step = owned.begin_step(job.job_id, name="vol-bench", argv=["quant"], idempotent=True)
    owned.attach_process(job.job_id, step.step_id, pid=11, pgid=9)
    assert owned.get(job.job_id).steps[0].pid_token == "tok"
    owned.request_cancel(job.job_id)
    done = owned.confirm_terminated(job.job_id)
    assert done.state == JobState.CANCELLED
    assert done.steps[0].state == JobState.CANCELLED
    running = owned.create(kind="turn", goal="g")
    owned.transition(running.job_id, JobState.RUNNING)
    with pytest.raises(IllegalTransition, match="confirm"):
        owned.confirm_terminated(running.job_id)


def test_begin_step_ids_attempts_and_job_stays_running_until_transition(
    ledger: JobLedger,
) -> None:
    run_id = new_run_id()
    job = ledger.create(kind="superpower", goal="g", run_id=run_id)
    step = ledger.begin_step(
        job.job_id, name="vol-bench", argv=["echo", "1"], idempotent=False, step_index=3
    )
    assert step.step_id == f"{run_id}.003"
    assert step.attempt == 1
    assert step.replayed is False
    assert step.state == JobState.RUNNING
    got = ledger.get(job.job_id)
    assert got.state == JobState.RUNNING
    assert got.revision == 2
    with pytest.raises(IllegalTransition, match="already exists"):
        ledger.begin_step(
            job.job_id, name="vol-bench", argv=["echo"], idempotent=False, step_index=3
        )
    second = ledger.begin_step(job.job_id, name="flash_retrieve", argv=[], idempotent=True)
    assert second.step_id == f"{run_id}.004"
    finished = ledger.finish_step(
        job.job_id,
        step.step_id,
        state=JobState.SUCCEEDED,
        exit_code=0,
        receipt_paths=["receipts/one.json"],
        detail="ok",
    )
    assert finished.exit_code == 0
    assert finished.receipt_paths == ["receipts/one.json"]
    assert ledger.get(job.job_id).state == JobState.RUNNING
    again = ledger.finish_step(
        job.job_id,
        step.step_id,
        state=JobState.SUCCEEDED,
        exit_code=9,
        detail="later",
    )
    assert again.detail == "ok" and again.exit_code == 0
    with pytest.raises(IllegalTransition):
        ledger.finish_step(job.job_id, step.step_id, state=JobState.FAILED, exit_code=1)
    assert ledger.get(job.job_id).steps[0].state == JobState.SUCCEEDED
    with pytest.raises(IllegalTransition):
        ledger.finish_step(job.job_id, step.step_id, state=JobState.RUNNING, exit_code=None)
    with pytest.raises(IllegalTransition, match="step not found"):
        ledger.attach_process(job.job_id, f"{run_id}.009", pid=3, pgid=None)
    with pytest.raises(ValueError):
        ledger.attach_process(job.job_id, step.step_id, pid=0, pgid=None)
    with pytest.raises(IllegalTransition, match="finished"):
        ledger.attach_process(job.job_id, step.step_id, pid=3, pgid=None)


def test_idempotency_key_skips_succeeded_work_and_blocks_inflight(ledger: JobLedger) -> None:
    job = ledger.create(kind="superpower", goal="g")
    step = ledger.begin_step(
        job.job_id,
        name="vol-bench",
        argv=["quant", "vol-bench"],
        idempotent=True,
        idempotency_key=" vol-1 ",
    )
    before = ledger.get(job.job_id)
    with pytest.raises(IllegalTransition, match="in flight"):
        ledger.begin_step(
            job.job_id, name="vol-bench", argv=["x"], idempotent=True, idempotency_key="vol-1"
        )
    assert len(ledger.get(job.job_id).steps) == 1
    ledger.finish_step(job.job_id, step.step_id, state=JobState.SUCCEEDED, exit_code=0)
    done = ledger.get(job.job_id)
    skipped = ledger.begin_step(
        job.job_id, name="vol-bench", argv=["x"], idempotent=True, idempotency_key="vol-1"
    )
    assert skipped.replayed is True
    assert skipped.attempt == 1
    assert skipped.step_id == step.step_id
    assert ledger.get(job.job_id).revision == done.revision
    assert len(ledger.get(job.job_id).steps) == 1
    other = ledger.create(kind="research", goal="elsewhere")
    cross = ledger.begin_step(
        other.job_id, name="vol-bench", argv=["x"], idempotent=True, idempotency_key="vol-1"
    )
    assert cross.replayed is True
    assert ledger.get(other.job_id).steps == []
    assert ledger.get(job.job_id).revision == done.revision
    found = ledger.find_step_by_key("vol-1")
    assert found is not None and found[0].job_id == job.job_id and found[1].replayed is False
    assert ledger.find_step_by_key("missing") is None
    assert ledger.find_step_by_key("   ") is None
    ledger.finish_step(job.job_id, step.step_id, state=JobState.SUCCEEDED, exit_code=0)
    failed = ledger.begin_step(
        other.job_id, name="vol-bench", argv=["x"], idempotent=False, idempotency_key="retry-me"
    )
    ledger.finish_step(other.job_id, failed.step_id, state=JobState.FAILED, exit_code=2)
    retried = ledger.begin_step(
        other.job_id, name="vol-bench", argv=["x"], idempotent=False, idempotency_key="retry-me"
    )
    assert retried.replayed is False and retried.attempt == 2
    assert len(ledger.get(other.job_id).steps) == 2
    assert before.revision == 2


def test_minted_run_id_and_resumable_steps_are_only_idempotent_interrupted(
    ledger: JobLedger,
) -> None:
    job = ledger.create(kind="superpower", goal="g")
    assert job.run_id is None
    safe = ledger.begin_step(job.job_id, name="vol-bench", argv=["a"], idempotent=True)
    risky = ledger.begin_step(job.job_id, name="flash_save", argv=["b"], idempotent=False)
    assert safe.step_id.endswith(".000")
    stored = ledger.get(job.job_id)
    assert stored.run_id == safe.step_id.rsplit(".", 1)[0]
    ledger.finish_step(job.job_id, safe.step_id, state=JobState.INTERRUPTED, exit_code=None)
    ledger.finish_step(job.job_id, risky.step_id, state=JobState.INTERRUPTED, exit_code=None)
    ledger.finish_step(
        job.job_id,
        ledger.begin_step(job.job_id, name="synthesize", argv=["c"], idempotent=True).step_id,
        state=JobState.SUCCEEDED,
        exit_code=0,
    )
    resumable = ledger.resumable_steps(job.job_id)
    assert [step.step_id for step in resumable] == [safe.step_id]
    assert resumable[0].idempotent is True
    ledger.transition(job.job_id, JobState.INTERRUPTED)
    with pytest.raises(IllegalTransition, match="cannot start"):
        ledger.begin_step(job.job_id, name="vol-bench", argv=["z"], idempotent=True)


def test_reconcile_queued_dead_orphan_mismatch_and_cancel(
    tmp_path: Path, clock: MutableClock
) -> None:
    probe = FakeProbe()
    killed: list[tuple[int, int | None]] = []

    def killer(pid: int, pgid: int | None) -> bool:
        killed.append((pid, pgid))
        return True

    root = tmp_path / "jobs"
    ledger = JobLedger(root, clock=clock, probe=probe, killer=killer)
    queued = ledger.create(kind="turn", goal="waiting")
    probe.fail = True
    report = ledger.reconcile()
    assert report.interrupted == (queued.job_id,)
    assert "had not started" in (ledger.get(queued.job_id).error or "")
    probe.fail = False

    dead_job = ledger.create(kind="superpower", goal="dead")
    dead_step = ledger.begin_step(
        dead_job.job_id, name="vol-bench", argv=["q"], idempotent=True, idempotency_key="dead"
    )
    other = ledger.begin_step(
        dead_job.job_id, name="flash_save", argv=["q"], idempotent=False, idempotency_key="side"
    )
    probe.tokens[70] = "a"
    probe.tokens[71] = "b"
    probe.alive[70] = False
    probe.alive[71] = False
    ledger.attach_process(dead_job.job_id, dead_step.step_id, pid=70, pgid=70)
    ledger.attach_process(dead_job.job_id, other.step_id, pid=71, pgid=None)
    ledger.request_cancel(dead_job.job_id)
    assert ledger.get(dead_job.job_id).state == JobState.CANCELLING
    report = ledger.reconcile()
    assert dead_job.job_id in report.interrupted
    recovered = ledger.get(dead_job.job_id)
    assert recovered.state == JobState.INTERRUPTED
    assert recovered.cancel_requested_at
    assert "not alive" in (recovered.error or "")
    assert {step.state for step in recovered.steps} == {JobState.INTERRUPTED}
    assert [step.step_id for step in ledger.resumable_steps(dead_job.job_id)] == [dead_step.step_id]
    assert killed == []

    probe.alive[80] = True
    probe.tokens[80] = "ours"
    live = ledger.create(kind="superpower", goal="live")
    live_step = ledger.begin_step(live.job_id, name="vol-bench", argv=["q"], idempotent=False)
    ledger.attach_process(live.job_id, live_step.step_id, pid=80, pgid=80)
    before = ledger.get(live.job_id).revision
    report = ledger.reconcile()
    assert report.orphan_running == (ProcessNote(80, live_step.step_id, live.job_id),)
    assert report.terminated == ()
    assert killed == []
    assert ledger.get(live.job_id).state == JobState.RUNNING
    assert ledger.get(live.job_id).revision == before

    probe.tokens[80] = "not-ours"
    report = ledger.reconcile()
    assert report.unverifiable == (ProcessNote(80, live_step.step_id, live.job_id),)
    assert report.orphan_running == ()
    assert ledger.get(live.job_id).state == JobState.RUNNING
    ledger.request_cancel(live.job_id)
    report = ledger.reconcile()
    assert killed == []
    assert report.terminated == ()
    assert ledger.get(live.job_id).state == JobState.CANCELLING

    probe.tokens[80] = "ours"
    report = ledger.reconcile()
    assert killed == [(80, 80)]
    assert report.orphan_running == (ProcessNote(80, live_step.step_id, live.job_id),)
    assert report.terminated == (ProcessNote(80, live_step.step_id, live.job_id),)
    kept = ledger.get(live.job_id)
    assert kept.state == JobState.CANCELLING
    assert kept.steps[0].detail == "reconcile: termination requested"
    assert kept.steps[0].state == JobState.RUNNING


def test_reconcile_without_killer_and_finished_steps_and_probe_errors(
    tmp_path: Path, clock: MutableClock
) -> None:
    probe = FakeProbe()
    ledger = JobLedger(tmp_path / "jobs", clock=clock, probe=probe)
    probe.alive[5] = True
    probe.tokens[5] = "t"
    job = ledger.create(kind="exec", goal="g")
    step = ledger.begin_step(job.job_id, name="vol-bench", argv=["q"], idempotent=False)
    ledger.attach_process(job.job_id, step.step_id, pid=5, pgid=5)
    ledger.request_cancel(job.job_id)
    report = ledger.reconcile()
    assert report.terminated == ()
    assert report.orphan_running[0].pid == 5
    assert ledger.get(job.job_id).state == JobState.CANCELLING

    done = ledger.create(kind="turn", goal="g")
    finished = ledger.begin_step(done.job_id, name="synthesize", argv=[], idempotent=True)
    ledger.finish_step(done.job_id, finished.step_id, state=JobState.SUCCEEDED, exit_code=0)
    report = ledger.reconcile()
    assert done.job_id in report.interrupted
    stored = ledger.get(done.job_id)
    assert stored.state == JobState.INTERRUPTED
    assert stored.steps[0].state == JobState.SUCCEEDED
    assert "no live process" in (stored.error or "")

    probe.fail = True
    shaky = ledger.create(kind="turn", goal="g")
    shaky_step = ledger.begin_step(shaky.job_id, name="vol-bench", argv=["q"], idempotent=False)
    ledger.attach_process(shaky.job_id, shaky_step.step_id, pid=5, pgid=None)
    report = ledger.reconcile()
    assert ProcessNote(5, shaky_step.step_id, shaky.job_id) in report.unverifiable
    assert ledger.get(shaky.job_id).state == JobState.RUNNING

    def boom(pid: int, pgid: int | None) -> bool:
        raise RuntimeError(f"secret {SECRET}")

    angry = FakeProbe()
    angry.alive[9] = True
    angry.tokens[9] = "t"
    hostile = JobLedger(tmp_path / "jobs2", clock=clock, probe=angry, killer=boom)
    owned = hostile.create(kind="exec", goal="g")
    owned_step = hostile.begin_step(owned.job_id, name="vol-bench", argv=["q"], idempotent=False)
    hostile.attach_process(owned.job_id, owned_step.step_id, pid=9, pgid=9)
    hostile.request_cancel(owned.job_id)
    report = hostile.reconcile()
    assert report.terminated == ()
    assert report.orphan_running[0].pid == 9
    assert SECRET not in repr(report)


def test_killer_false_does_not_count_as_terminated(tmp_path: Path, clock: MutableClock) -> None:
    probe = FakeProbe()
    probe.alive[3] = True
    probe.tokens[3] = "t"

    def killer(pid: int, pgid: int | None) -> bool:
        return False

    ledger = JobLedger(tmp_path / "jobs", clock=clock, probe=probe, killer=killer)
    job = ledger.create(kind="exec", goal="g")
    step = ledger.begin_step(job.job_id, name="vol-bench", argv=["q"], idempotent=False)
    ledger.attach_process(job.job_id, step.step_id, pid=3, pgid=3)
    ledger.request_cancel(job.job_id)
    report = ledger.reconcile()
    assert report.orphan_running == (ProcessNote(3, step.step_id, job.job_id),)
    assert report.terminated == ()
    assert ledger.get(job.job_id).steps[0].detail == ""


def test_corrupt_files_are_quarantined_by_list_get_and_reconcile(
    ledger: JobLedger, tmp_path: Path
) -> None:
    job = ledger.create(kind="turn", goal="good")
    ledger.transition(job.job_id, JobState.RUNNING)
    ledger.transition(job.job_id, JobState.SUCCEEDED)
    root = tmp_path / "jobs"
    garbage_id = new_job_id()
    garbage = root / f"{garbage_id}.json"
    garbage.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreCorruptError):
        ledger.transition(garbage_id, JobState.RUNNING)
    assert garbage.read_text(encoding="utf-8") == "{not json"
    with pytest.raises(StoreCorruptError):
        ledger.get(garbage_id)
    assert not garbage.exists()
    assert any(path.name.startswith(garbage.name) for path in root.iterdir())

    bad_id = new_job_id()
    bad = root / f"{bad_id}.json"
    bad.write_text(json.dumps({"schema_version": 99, "job_id": bad_id}), encoding="utf-8")
    found = ledger.list()
    assert [item.job_id for item in found] == [job.job_id]
    assert not bad.exists()
    report = ledger.reconcile()
    assert report.interrupted == ()
    assert report.corrupt == ()

    mismatched = ledger.create(kind="turn", goal="copy")
    path = root / f"{mismatched.job_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = ledger.reconcile()
    assert path.name in report.corrupt
    assert not path.exists()
    assert ledger.get(job.job_id).state == JobState.SUCCEEDED

    if os.name == "posix":
        link_id = new_job_id()
        link = root / f"{link_id}.json"
        link.symlink_to(root / f"{job.job_id}.json")
        with pytest.raises(StoreCorruptError, match="symlink"):
            ledger.get(link_id)
        assert not link.exists()
        assert (root / f"{job.job_id}.json").is_file()


def test_list_orders_filters_and_limits(ledger: JobLedger, clock: MutableClock) -> None:
    assert ledger.list() == []
    assert ledger.reconcile() == ReconcileReport((), (), (), (), ())
    ids: list[str] = []
    for _ in range(3):
        ids.append(ledger.create(kind="turn", goal="g").job_id)
        clock.moment += timedelta(seconds=1)
    assert [job.job_id for job in ledger.list()] == list(reversed(ids))
    assert [job.job_id for job in ledger.list(limit=2)] == list(reversed(ids))[:2]
    assert ledger.list(limit=0) == []
    with pytest.raises(ValueError):
        ledger.list(limit=-1)
    ledger.transition(ids[0], JobState.RUNNING)
    running = ledger.list(states={JobState.RUNNING})
    assert [job.job_id for job in running] == [ids[0]]
    assert {job.job_id for job in ledger.list(states=[JobState.QUEUED])} == {ids[1], ids[2]}


def test_prune_respects_age_and_never_drops_nonterminal(
    ledger: JobLedger, clock: MutableClock
) -> None:
    assert ledger.prune() == 0
    fresh = ledger.create(kind="turn", goal="young")
    ledger.transition(fresh.job_id, JobState.RUNNING)
    ledger.transition(fresh.job_id, JobState.SUCCEEDED)
    created = datetime.fromisoformat(fresh.created_at)
    clock.moment = created + timedelta(days=59)
    assert ledger.prune(max_jobs=10, max_age_days=60) == 0
    clock.moment = created + timedelta(days=60)
    assert ledger.prune(max_jobs=10, max_age_days=60) == 1
    with pytest.raises(JobNotFound):
        ledger.get(fresh.job_id)

    clock.moment = datetime(2026, 2, 1, tzinfo=UTC)
    running = ledger.create(kind="research", goal="keep")
    terminal_ids: list[str] = []
    for _ in range(4):
        clock.moment += timedelta(hours=1)
        job = ledger.create(kind="turn", goal="old")
        ledger.transition(job.job_id, JobState.RUNNING)
        ledger.transition(job.job_id, JobState.FAILED, error="x")
        terminal_ids.append(job.job_id)
    clock.moment += timedelta(days=1)
    removed = ledger.prune(max_jobs=3, max_age_days=1000)
    assert removed == 2
    remaining = {job.job_id for job in ledger.list(limit=10)}
    assert running.job_id in remaining
    assert remaining - {running.job_id} == set(terminal_ids[2:])
    # Oldest two terminal jobs are gone; the non-terminal job is kept even if over age.
    clock.moment += timedelta(days=5000)
    assert ledger.prune(max_jobs=0, max_age_days=1) == 2
    assert {job.job_id for job in ledger.list(limit=10)} == {running.job_id}
    with pytest.raises(ValueError):
        ledger.prune(max_jobs=-1)
    with pytest.raises(ValueError):
        ledger.prune(max_age_days=True)  # type: ignore[arg-type]


def test_concurrent_steps_and_idempotency(ledger: JobLedger) -> None:
    job = ledger.create(kind="superpower", goal="g")
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            ledger.begin_step(
                job.job_id,
                name="vol-bench",
                argv=["echo", str(index)],
                idempotent=True,
                idempotency_key=f"k-{index}",
            )
        except BaseException as exc:  # noqa: BLE001 — surface thread failures
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert errors == []
    stored = ledger.get(job.job_id)
    assert len(stored.steps) == 8
    assert len({step.step_id for step in stored.steps}) == 8
    assert stored.revision == 9

    done = ledger.finish_step(
        job.job_id, stored.steps[0].step_id, state=JobState.SUCCEEDED, exit_code=0
    )
    key = stored.steps[0].idempotency_key
    assert key is not None
    revision = ledger.get(job.job_id).revision
    barrier = threading.Barrier(6)
    replays: list[bool] = []

    def replay() -> None:
        barrier.wait(5)
        got = ledger.begin_step(
            job.job_id, name="vol-bench", argv=["echo"], idempotent=True, idempotency_key=key
        )
        replays.append(got.replayed and got.step_id == done.step_id)

    threads = [threading.Thread(target=replay) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert replays == [True] * 6
    assert ledger.get(job.job_id).revision == revision
    assert len(ledger.get(job.job_id).steps) == 8

    contested = ledger.create(kind="research", goal="g")
    barrier = threading.Barrier(8)
    outcomes: list[str] = []
    guard = threading.Lock()

    def once() -> None:
        barrier.wait(5)
        try:
            ledger.begin_step(
                contested.job_id,
                name="vol-bench",
                argv=["echo"],
                idempotent=False,
                idempotency_key="same",
            )
        except IllegalTransition:
            kind = "blocked"
        except Exception as exc:  # noqa: BLE001
            kind = f"error:{type(exc).__name__}"
        else:
            kind = "started"
        with guard:
            outcomes.append(kind)

    threads = [threading.Thread(target=once) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert outcomes.count("started") == 1
    assert outcomes.count("blocked") == 7
    assert not any(item.startswith("error:") for item in outcomes)
    assert len(ledger.get(contested.job_id).steps) == 1


def test_system_probe_platforms(tmp_path: Path) -> None:
    start = "424242"
    tail = " ".join(["S", *["0"] * 18, start, "99"])
    live_dir = tmp_path / "7"
    live_dir.mkdir()
    (live_dir / "stat").write_text(f"7 (python (worker)) {tail}\n", encoding="utf-8")
    zombie = tmp_path / "8"
    zombie.mkdir()
    (zombie / "stat").write_text(
        "8 (zsh) " + " ".join(["Z", *["0"] * 18, "11"]) + "\n", encoding="utf-8"
    )
    broken = tmp_path / "5"
    broken.mkdir()
    (broken / "stat").write_text("nosep\n", encoding="utf-8")
    probe = SystemProcessProbe(proc_root=tmp_path, platform="linux")
    assert probe.is_alive(7) is True and probe.token(7) == start
    assert probe.is_alive(8) is False and probe.token(8) == "11"
    assert probe.is_alive(5) is True and probe.token(5) is None
    assert probe.is_alive(99) is False and probe.token(99) is None
    assert probe.is_alive(0) is False and probe.token(-1) is None
    if sys.platform == "linux":
        current = SystemProcessProbe()
        assert current.is_alive(os.getpid()) is True
        token = current.token(os.getpid())
        assert token is not None and token.isdigit()
        missing = 2_000_000_000
        if not (Path("/proc") / str(missing)).exists():
            assert current.is_alive(missing) is False

    def ps(argv: list[str]) -> tuple[int, str]:
        if "lstart=" in argv:
            return 0, "Mon Oct 12 03:00:00 2026\n"
        return 0, "4\n"

    darwin = SystemProcessProbe(platform="darwin", runner=ps)
    assert darwin.is_alive(4) is True
    assert darwin.token(4) == "Mon Oct 12 03:00:00 2026"
    quiet = SystemProcessProbe(platform="darwin", runner=lambda argv: (1, ""))
    assert quiet.is_alive(4) is False and quiet.token(4) is None

    def tasklist(argv: list[str]) -> tuple[int, str]:
        return 0, '"python.exe","44","Console"\n'

    windows = SystemProcessProbe(platform="win32", runner=tasklist)
    assert windows.is_alive(44) is True
    assert windows.token(44) is None
    empty = SystemProcessProbe(
        platform="win32",
        runner=lambda argv: (0, "INFO: No tasks are running matching the specified criteria.\n"),
    )
    assert empty.is_alive(44) is False

    def broken_runner(argv: list[str]) -> tuple[int, str]:
        raise OSError("no tasklist")

    with pytest.raises(OSError):
        SystemProcessProbe(platform="win32", runner=broken_runner).is_alive(4)


def test_terminate_process_is_posix_sigterm_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, int]] = []

    def killpg(pgid: int, sig: int) -> None:
        calls.append(("pg", pgid, sig))

    def kill(pid: int, sig: int) -> None:
        calls.append(("pid", pid, sig))

    monkeypatch.setattr(jobs_mod.os, "killpg", killpg)
    monkeypatch.setattr(jobs_mod.os, "kill", kill)
    monkeypatch.setattr(jobs_mod.sys, "platform", "linux")
    import signal

    assert terminate_process(12, 34) is True
    assert calls == [("pg", 34, signal.SIGTERM)]
    assert terminate_process(12, None) is True
    assert calls[-1] == ("pid", 12, signal.SIGTERM)

    def fail(pid: int, sig: int) -> None:
        raise OSError("no such process")

    monkeypatch.setattr(jobs_mod.os, "kill", fail)
    assert terminate_process(12, None) is False
    monkeypatch.setattr(jobs_mod.sys, "platform", "win32")

    def forbidden(pid: int, sig: int) -> None:
        raise AssertionError("windows must not signal")

    monkeypatch.setattr(jobs_mod.os, "kill", forbidden)
    assert terminate_process(12, 34) is False
    assert terminate_process(True, None) is False  # type: ignore[arg-type]
    assert terminate_process(0, None) is False
