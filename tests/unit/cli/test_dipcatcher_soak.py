"""Hermetic checks for the bounded soak harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load():
    path = ROOT / "scripts" / "dipcatcher_soak.py"
    spec = importlib.util.spec_from_file_location("dipcatcher_soak", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load dipcatcher_soak")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOAK = _load()


def _sample(**kwargs: object) -> object:
    base = {
        "fd_count": 10,
        "nondaemon_threads": 1,
        "daemon_threads": ("fx1-stub-model",),
        "traced_bytes": 1000,
        "child_pids": (),
        "registry_pids": (),
    }
    base.update(kwargs)
    return SOAK.ResourceSample(**base)


def test_scope_denies_the_24_hour_soak() -> None:
    assert "NOT" in SOAK.SCOPE
    assert "24-hour" in SOAK.SCOPE
    assert "this machine" in SOAK.SCOPE


def test_schedule_is_deterministic_and_covers_every_kind() -> None:
    first = SOAK.schedule(7)
    assert first == SOAK.schedule(7)
    assert first != SOAK.schedule(8)
    assert set(first) == set(SOAK.WORKLOADS)


def test_drive_always_completes_one_pass() -> None:
    calls: list[str] = []

    def clock() -> float:
        return 100.0 if calls else 0.0

    def run_one(kind: str) -> object:
        calls.append(kind)
        return SOAK.CycleResult(kind, "ok", True, "")

    out = SOAK.drive_until(0.0, ["a", "b", "c"], clock, run_one)
    assert [item.kind for item in out] == ["a", "b", "c"]


def test_drive_stops_after_duration_once_the_pass_is_done() -> None:
    ticks = {"n": 0}

    def clock() -> float:
        ticks["n"] += 1
        return 0.0 if ticks["n"] == 1 else 50.0

    def run_one(kind: str) -> object:
        return SOAK.CycleResult(kind, "ok", True, "")

    out = SOAK.drive_until(10.0, ["a", "b"], clock, run_one, max_cycles=10)
    assert [item.kind for item in out] == ["a", "b"]


def test_proc_children_and_ps_fallback(tmp_path: Path) -> None:
    root = tmp_path / "proc"
    (root / "4").mkdir(parents=True)
    (root / "9").mkdir()
    (root / "4" / "status").write_text("Name:\tx\nPPid:\t1\n", encoding="utf-8")
    (root / "9" / "status").write_text("PPid:\t4\n", encoding="utf-8")
    assert SOAK.child_pids(4, proc_root=root) == (9,)
    missing = tmp_path / "absent"
    assert SOAK.child_pids(5, proc_root=missing, ps_runner=lambda: "  9  5\n  3  1\n") == (9,)
    assert SOAK.child_pids(5, proc_root=missing, ps_runner=lambda: None) is None


def test_count_fds_missing_dir_is_unavailable(tmp_path: Path) -> None:
    assert SOAK.count_fds(tmp_path / "no-fd") is None
    fd_root = tmp_path / "fd"
    fd_root.mkdir()
    (fd_root / "0").write_text("", encoding="utf-8")
    (fd_root / "1").write_text("", encoding="utf-8")
    assert SOAK.count_fds(fd_root) == 2


def test_leaking_fake_child_fails_the_checker() -> None:
    class Leak:
        def live(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(pid=4242)]

    owned = SOAK.registry_pids(Leak())
    problems = SOAK.leak_problems((4242,), owned)
    assert problems
    assert any("4242" in item for item in problems)
    assert SOAK.leak_problems((), ()) == ()


def test_registry_without_live_fails() -> None:
    assert SOAK.registry_pids(object()) == (-1,)
    assert SOAK.leak_problems((), SOAK.registry_pids(object()))


def test_request_log_trim_keeps_a_short_tail() -> None:
    server = SimpleNamespace(_requests=list(range(20)))
    SOAK.trim_request_log(server, keep=4)
    assert server._requests == [16, 17, 18, 19]
    SOAK.trim_request_log(server, keep=4)
    assert server._requests == [16, 17, 18, 19]
    SOAK.trim_request_log(None)


def test_diff_flags_fd_threads_and_alloc() -> None:
    baseline = _sample()
    grown = _sample(
        fd_count=50, nondaemon_threads=3, traced_bytes=5000, daemon_threads=("Thread-9",)
    )
    tight = SOAK.Bounds(fd_growth=2, tracemalloc_growth_bytes=100)
    problems = SOAK.diff_resources(baseline, grown, tight, check_alloc=True)
    assert any(item.startswith("fd count") for item in problems)
    assert any("non-daemon" in item for item in problems)
    assert any("unexpected daemon" in item for item in problems)
    assert any("tracemalloc" in item for item in problems)


def test_named_stub_daemon_within_cap_passes() -> None:
    baseline = _sample(traced_bytes=1000)
    current = _sample(daemon_threads=("fx1-stub-model", "fx1-stub-handler"), traced_bytes=1000)
    assert SOAK.diff_resources(current, current, SOAK.Bounds(), check_alloc=False) == ()
    assert (
        SOAK.diff_resources(baseline, current, SOAK.Bounds(daemon_cap=1), check_alloc=False) != ()
    )


def test_steady_growth_fails_when_later_cycles_keep_allocating() -> None:
    assert SOAK.steady_growth_problem([100, 66_000_000], 2, 1000) is None
    problem = SOAK.steady_growth_problem([100, 200, 50_000_000], 2, 1000)
    assert problem is not None
    assert "after the first pass" in problem


def test_alloc_growth_is_ignored_when_check_is_off() -> None:
    baseline = _sample(traced_bytes=10)
    current = _sample(traced_bytes=10_000_000)
    assert SOAK.diff_resources(baseline, current, SOAK.Bounds(), check_alloc=False) == ()


def test_job_views_reject_nonterminal_and_bare_interrupt() -> None:
    clean = [
        {"job_id": "j1", "state": "succeeded", "error": "", "steps": []},
        {"job_id": "j2", "state": "interrupted", "error": "reconcile: dead", "steps": []},
        {
            "job_id": "j3",
            "state": "cancelled",
            "error": "",
            "steps": [{"state": "cancelled", "detail": ""}],
        },
    ]
    assert SOAK.audit_job_views(clean) == ()
    leaked = [{"job_id": "j4", "state": "running", "error": "", "steps": []}]
    assert SOAK.audit_job_views(leaked)
    bare = [{"job_id": "j5", "state": "interrupted", "error": "", "steps": []}]
    assert any("without a reason" in item for item in SOAK.audit_job_views(bare))
    step = [
        {
            "job_id": "j6",
            "state": "succeeded",
            "error": "",
            "steps": [{"state": "running", "detail": ""}],
        }
    ]
    assert any("step" in item for item in SOAK.audit_job_views(step))


def test_real_ledger_queued_job_fails(tmp_path: Path) -> None:
    from fx1.interactive.jobs import JobLedger

    ledger = JobLedger(tmp_path)
    record = ledger.create(kind="soak", goal="left queued on purpose")
    views = [
        {
            "job_id": record.job_id,
            "state": str(record.state),
            "error": record.error or "",
            "steps": [],
        }
    ]
    assert SOAK.audit_job_views(views)


def test_sessions_reject_bad_json_and_running_jobs(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    (tmp_path / "run.json").write_text(
        json.dumps({"id": "abc", "job": {"state": "running"}}), encoding="utf-8"
    )
    (tmp_path / "ok.json").write_text(
        json.dumps({"id": "def", "job": {"state": "interrupted", "resume": "manual"}}),
        encoding="utf-8",
    )
    problems = SOAK.audit_sessions(tmp_path)
    assert any("invalid JSON" in item for item in problems)
    assert any("still running" in item for item in problems)
    assert not any("ok.json" in item for item in problems)


def test_secret_scan_fails_when_the_token_is_planted(tmp_path: Path) -> None:
    secret = SOAK.planted_secret(7)
    (tmp_path / "notes.txt").write_text(f"leaked {secret}\n", encoding="utf-8")
    assert SOAK.secret_hits([tmp_path], secret)
    assert SOAK.secret_hits([tmp_path], SOAK.planted_secret(8)) == ()


def test_fault_matrix_and_deliberate_misses() -> None:
    assert SOAK.fault_recovered("fault_401", "error")
    assert not SOAK.fault_recovered("fault_401", "ok")
    assert SOAK.fault_recovered("fault_timeout", "budget")
    assert SOAK.fault_recovered("cancel_turn", "cancelled")
    assert not SOAK.fault_recovered("approval_denial", "ok", "ran train")
    assert SOAK.fault_recovered("approval_denial", "ok", "train: not approved")
    assert not SOAK.fault_recovered("superpower", "ok", "SYNTHETIC only")
    assert SOAK.fault_recovered("superpower", "ok", "lane-power ran DATA_LABEL=SYNTHETIC")
    assert not SOAK.fault_recovered("cancel_job", "ok", "stopped")
    assert SOAK.fault_recovered("cancel_job", "cancelled", "stopped; child observed")
    assert not SOAK.fault_recovered("unknown", "ok")


def test_lane_power_plan_is_minimum_synthetic(tmp_path: Path) -> None:
    payload = json.loads(SOAK.lane_power_plan(tmp_path / "out"))
    step = payload["steps"][0]
    assert step["capability"] == "lane-power"
    assert step["args"]["--n-steps"] == 16
    assert step["args"]["--n-seeds"] == 1
    assert step["args"]["--lanes"] == "serial_watch"
    assert "SYNTHETIC" in step["why"]
    assert payload["rationale"].startswith("SYNTHETIC")


def test_build_record_schema_and_failure_flag() -> None:
    clean = SOAK.build_record(
        seed=3,
        duration_s=8,
        elapsed_s=8.5,
        warmup_s=0.1,
        cycles=(SOAK.CycleResult("chat_fx1", "ok", True, "pong"),),
        failures=(),
        resources={"orphan_children": [], "registry_live": []},
        jobs=(),
        session_problems=(),
        secret_found=(),
        bounds=SOAK.Bounds(),
    )
    assert clean["schema"] == "dipcatcher.soak.v1"
    assert clean["scope"] == SOAK.SCOPE
    assert clean["reference_24h"] is False
    assert clean["ok"] is True
    assert clean["counts"]["chat_fx1"] == 1
    assert clean["counts"]["superpower"] == 0
    assert clean["bench"]["evidence_class"] == "SYNTHETIC"
    dirty = SOAK.build_record(
        seed=3,
        duration_s=8,
        elapsed_s=1,
        warmup_s=0,
        cycles=(),
        failures=("orphan child pid 7",),
        resources={"orphan_children": [7], "registry_live": []},
        jobs=(),
        session_problems=(),
        secret_found=(),
        bounds=SOAK.Bounds(),
    )
    assert dirty["ok"] is False


def test_write_record_is_private_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "soak.json"
    SOAK.write_record(path, {"schema": SOAK.SCHEMA, "scope": SOAK.SCOPE, "ok": True})
    assert path.stat().st_mode & 0o777 == 0o600
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema"] == SOAK.SCHEMA


def test_usage_error_and_duration_guard() -> None:
    with pytest.raises(SOAK.SoakUsageError):
        SOAK._require_duration(0)
    with pytest.raises(SOAK.SoakUsageError):
        SOAK._require_duration(86401)
    assert SOAK.main(["--duration-s", "0"]) == 2


def test_real_bounded_soak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config"
    sessions = tmp_path / "sessions"
    work = tmp_path / "work"
    out = tmp_path / "soak.json"
    for path in (config, sessions, work):
        path.mkdir()
    monkeypatch.setenv("FX1_CONFIG_DIR", str(config))
    monkeypatch.setenv("FX1_SESSION_DIR", str(sessions))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    for name in ("MOONSHOT_API_KEY", "FX1_BASE_URL", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
    code = SOAK.main(
        [
            "--duration-s",
            "8",
            "--seed",
            "7",
            "--out",
            str(out),
            "--config-dir",
            str(config),
            "--session-dir",
            str(sessions),
            "--work-dir",
            str(work),
        ]
    )
    text = out.read_text(encoding="utf-8") if out.exists() else ""
    assert code == 0, text[:4000]
    record = json.loads(text)
    assert record["schema"] == "dipcatcher.soak.v1"
    assert record["ok"] is True
    assert record["failures"] == []
    assert record["orphan_children"] == []
    assert record["registry_live"] == []
    assert record["secret_hits"] == []
    assert record["session_problems"] == []
    assert "24-hour" in record["scope"]
    assert record["reference_24h"] is False
    assert out.stat().st_mode & 0o777 == 0o600
    for kind in SOAK.WORKLOADS:
        assert record["counts"][kind] >= 1, kind
    assert SOAK.planted_secret(7).encode() not in text.encode()
