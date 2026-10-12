"""Live durability: real processes, files, crashes, and hostile inputs."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fx1.interactive import atomicstore
from fx1.interactive import jobs as jobs_mod
from fx1.interactive.atomicstore import (
    FileLock,
    StoreCorruptError,
    StoreError,
    atomic_write_json,
    quarantine,
    read_json,
)
from fx1.interactive.budgets import BudgetError, RunBudget, merge_budget, merge_tool_budget
from fx1.interactive.jobs import JobLedger, JobState, SystemProcessProbe
from fx1.interactive.runctx import RunContext, is_valid_id, new_session_id
from fx1.interactive.sessions import SessionBusy, SessionError, SessionStore
from fx1.interactive.usage_budget import UsageLimits, UsageTracker

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX crash and mode tests")


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sessions"))


def _py(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    return subprocess.run(
        [sys.executable, "-c", *args],
        check=False,
        capture_output=True,
        text=True,
        env=proc_env,
        timeout=20,
    )


def _stat_text(start: str, state: str = "S") -> str:
    tail = " ".join([state, *["0"] * 18, start])
    return f"7 (python) {tail}\n"


def test_atomic_writes_survive_sigkill(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    code = (
        "import sys; from pathlib import Path; "
        "from fx1.interactive.atomicstore import atomic_write_json\n"
        "path = Path(sys.argv[1]); i = 0\n"
        "while True:\n"
        "    atomic_write_json(path, {'i': i, 'blob': 'x' * 200}); i += 1\n"
    )
    for n in range(30):
        proc = subprocess.Popen([sys.executable, "-c", code, str(path)])
        time.sleep(0.002 + (n % 5) * 0.002)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)
        if path.exists():
            parsed = json.loads(path.read_text(encoding="utf-8"))
            assert set(parsed) == {"i", "blob"}
            assert parsed["blob"] == "x" * 200
        with FileLock(path, timeout_s=2):
            pass
        leftovers = [item.name for item in path.parent.iterdir() if item.suffix == ".json"]
        assert leftovers == [path.name] or not path.exists()


def test_quarantine_keeps_every_payload_and_nan_is_not_json(tmp_path: Path) -> None:
    first = tmp_path / "state.json"
    first.write_text("one", encoding="utf-8")
    original = atomicstore.time.time
    atomicstore.time.time = lambda: 1000
    try:
        kept = quarantine(first)
        first.write_text("two", encoding="utf-8")
        second = quarantine(first)
    finally:
        atomicstore.time.time = original
    assert kept is not None and second is not None and kept != second
    assert kept.read_text(encoding="utf-8") == "one"
    assert second.read_text(encoding="utf-8") == "two"
    path = tmp_path / "nan.json"
    atomic_write_json(path, {"ok": 1})
    with pytest.raises(ValueError):
        atomic_write_json(path, {"x": float("nan")})
    assert read_json(path) == {"ok": 1}
    path.write_text('{"x": NaN}\n', encoding="utf-8")
    with pytest.raises(StoreCorruptError):
        read_json(path)


def test_private_dir_is_tightened_and_symlink_is_refused(tmp_path: Path) -> None:
    loose = tmp_path / "loose"
    loose.mkdir()
    os.chmod(loose, 0o755)
    atomic_write_json(loose / "a.json", {"k": 1})
    assert stat.S_IMODE(loose.stat().st_mode) == 0o700
    assert stat.S_IMODE((loose / "a.json").stat().st_mode) == 0o600
    link = tmp_path / "link"
    link.symlink_to(loose, target_is_directory=True)
    with pytest.raises(StoreError, match="symlink"):
        atomic_write_json(link / "b.json", {"k": 2})


def test_windows_lock_prepares_a_byte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Msvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 0

        def locking(self, fd: int, mode: int, nbytes: int) -> None:
            del mode
            if os.fstat(fd).st_size < nbytes:
                raise OSError(13, "Permission denied")

    monkeypatch.setitem(sys.modules, "msvcrt", _Msvcrt())
    monkeypatch.setattr(sys, "platform", "win32")
    with FileLock(tmp_path / "ledger.json", timeout_s=0.3, poll_s=0.01):
        assert (tmp_path / "ledger.json.lock").stat().st_size >= 1


def test_update_json_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "counter.json"
    code = (
        "import sys; from pathlib import Path; "
        "from fx1.interactive.atomicstore import update_json\n"
        "path, n = Path(sys.argv[1]), int(sys.argv[2])\n"
        "for _ in range(n):\n"
        "    update_json(path, lambda cur: {'n': int(cur['n']) + 1}, "
        "default={'n': 0}, lock_timeout_s=10)\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code, str(path), "15"]) for _ in range(4)]
    for proc in procs:
        assert proc.wait(timeout=20) == 0
    assert read_json(path) == {"n": 60}


def test_session_permissions_precedence_and_hostile_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("FX1_SESSION_DIR", str(tmp_path / "sess"))
    store = SessionStore()
    record = store.create(model="fx1", messages=[{"role": "user", "content": "hello"}])
    directory = tmp_path / "sess"
    path = directory / f"{record['id']}.json"
    assert path.is_file()
    assert not any((tmp_path / "cfg").rglob("*.json"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    loose = tmp_path / "loose-sessions"
    loose.mkdir()
    os.chmod(loose, 0o755)
    SessionStore(loose).create(model="fx1-lite")
    assert stat.S_IMODE(loose.stat().st_mode) == 0o700
    for bad in ("../x", "/tmp/" + "a" * 32, "a" * 31 + "\x00", "é" * 16, "A" * 32):
        with pytest.raises(SessionError):
            store.load(bad)


def test_session_symlink_schema_and_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SessionError, match="symlink"):
        SessionStore(link).create(model="fx1")
    assert list(real.iterdir()) == []

    store = SessionStore(tmp_path / "sessions")
    created = store.create(
        model="fx1",
        messages=[{"role": "user", "content": "see sk-123456789 please"}],
    )
    raw = (store.directory / f"{created['id']}.json").read_text(encoding="utf-8")
    assert "sk-123456789" not in raw
    assert "[redacted]" in raw
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    kept = store.create(model="fx1", messages=[{"role": "user", "content": "false"}])
    assert kept["messages"][0]["content"] == "false"

    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps(
            {
                "schema": 1,
                "id": created["id"],
                "model": "fx1",
                "messages": [{"role": "user", "content": "LEAKME-OUTSIDE"}],
                "created_at": "2020-01-01T00:00:00+00:00",
                "updated_at": "2020-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    session_path = store.directory / f"{created['id']}.json"
    session_path.unlink()
    session_path.symlink_to(outside)
    with pytest.raises(SessionError, match="symlink"):
        store.load(created["id"])
    assert "LEAKME-OUTSIDE" in outside.read_text(encoding="utf-8")

    newer = "ab" * 16
    newer_path = store.directory / f"{newer}.json"
    newer_path.parent.mkdir(parents=True, exist_ok=True)
    newer_path.write_text(
        json.dumps(
            {
                "schema": 2,
                "id": newer,
                "model": "fx1",
                "messages": [{"role": "user", "content": "KEEP"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SessionError, match="newer"):
        store.save({"schema": 1, "id": newer, "model": "fx1", "messages": [], "revision": 0})
    assert json.loads(newer_path.read_text(encoding="utf-8"))["schema"] == 2
    assert json.loads(newer_path.read_text(encoding="utf-8"))["messages"][0]["content"] == "KEEP"

    older = "cd" * 16
    older_path = store.directory / f"{older}.json"
    older_path.write_text(
        json.dumps(
            {
                "schema": 0,
                "id": older,
                "model": "fx1-lite",
                "messages": [{"role": "user", "content": "old"}],
            }
        ),
        encoding="utf-8",
    )
    loaded = store.load(older)
    assert loaded["schema"] == 1
    assert loaded["messages"][0]["content"] == "old"
    assert loaded["job"]["state"] == "idle"
    store.save(loaded)
    assert json.loads(older_path.read_text(encoding="utf-8"))["schema"] == 1


def test_corrupt_sessions_are_reported_and_large_text_roundtrips(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    good = store.create(model="fx1")
    bad = "ef" * 16
    (store.directory / f"{bad}.json").write_bytes(b"\xff\xfe not json")
    notes = store.diagnose()
    assert any(bad in note for note in notes)
    assert store.latest() is not None
    assert store.latest()["id"] == good["id"]
    with pytest.raises(SessionError):
        store.load(bad)
    blob = "q" * 500_000
    started = time.perf_counter()
    huge = store.create(model="fx1", messages=[{"role": "user", "content": blob}])
    assert store.load(huge["id"])["messages"][0]["content"] == blob
    assert time.perf_counter() - started < 3.0


def test_session_writers_do_not_lose_updates_and_reserve_is_exclusive(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    record = store.create(model="fx1")
    sid = record["id"]
    code = (
        "import sys, time; from pathlib import Path\n"
        "from fx1.interactive.sessions import SessionStore, SessionBusy\n"
        "root, sid, label, flagdir = sys.argv[1:]\n"
        "store = SessionStore(Path(root)); rec = store.load(sid)\n"
        "Path(flagdir, label).write_text('loaded')\n"
        "deadline = time.time() + 5\n"
        "while time.time() < deadline:\n"
        "    if (Path(flagdir) / 'A').exists() and (Path(flagdir) / 'B').exists():\n"
        "        break\n"
        "    time.sleep(0.01)\n"
        "rec['messages'].append({'role': 'user', 'content': label})\n"
        "try:\n"
        "    store.save(rec); print('saved')\n"
        "except SessionBusy:\n"
        "    print('busy')\n"
    )
    flags = tmp_path / "flags"
    flags.mkdir()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(store.directory), sid, label, str(flags)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for label in ("A", "B")
    ]
    outputs = []
    for proc in procs:
        out, _err = proc.communicate(timeout=20)
        assert proc.returncode == 0
        outputs.append(out.strip())
    assert sorted(outputs) == ["busy", "saved"]
    assert len(store.load(sid)["messages"]) == 1

    append = (
        "import sys; from pathlib import Path\n"
        "from fx1.interactive.sessions import SessionStore\n"
        "store = SessionStore(Path(sys.argv[1])); sid, label = sys.argv[2], sys.argv[3]\n"
        "for i in range(8):\n"
        "    store.append_turn(sid, {'role': 'user', 'content': f'{label}-{i}'})\n"
        "print('ok')\n"
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", append, str(store.directory), sid, label],
            stdout=subprocess.PIPE,
            text=True,
        )
        for label in ("p", "q")
    ]
    for proc in procs:
        assert proc.wait(timeout=20) == 0
    contents = {item["content"] for item in store.load(sid)["messages"]}
    assert {f"p-{i}" for i in range(8)} <= contents
    assert {f"q-{i}" for i in range(8)} <= contents

    hold = (
        "import sys; from pathlib import Path\n"
        "from fx1.interactive.sessions import SessionStore\n"
        "store = SessionStore(Path(sys.argv[1]))\n"
        "with store.reserve(sys.argv[2]):\n"
        "    print('held', flush=True)\n"
        "    sys.stdin.readline()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", hold, str(store.directory), sid],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None and child.stdin is not None
    assert child.stdout.readline().strip() == "held"
    started = time.perf_counter()
    with pytest.raises(SessionBusy, match="busy"), store.reserve(sid):
        pass
    assert time.perf_counter() - started < 1.0
    child.stdin.write("\n")
    child.stdin.flush()
    assert child.wait(timeout=5) == 0


def test_session_crash_leaves_parseable_files(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    record = store.create(model="fx1")
    code = (
        "import sys; from pathlib import Path\n"
        "from fx1.interactive.sessions import SessionStore\n"
        "store = SessionStore(Path(sys.argv[1])); sid = sys.argv[2]; i = 0\n"
        "while True:\n"
        "    store.append_turn(sid, {'role': 'user', 'content': f'm-{i}-' + ('y' * 40)})\n"
        "    i += 1\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(store.directory), record["id"]])
    time.sleep(0.05)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=5)
    for path in store.directory.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    assert store.latest() is not None
    with FileLock(store.directory / f"{record['id']}.json", timeout_s=2):
        pass


def test_probe_does_not_pass_secrets_to_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-" + "z" * 24)
    captured: dict[str, dict[str, str]] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = kwargs.get("env")
        assert isinstance(env, dict)
        captured["env"] = env
        return subprocess.CompletedProcess(argv, 0, stdout="4\n", stderr="")

    monkeypatch.setattr(jobs_mod.subprocess, "run", fake_run)
    assert SystemProcessProbe(platform="darwin").is_alive(4) is True
    assert "MOONSHOT_API_KEY" not in captured["env"]
    assert captured["env"]["LC_ALL"] == "C"


def test_unknown_job_fields_roundtrip_and_bool_schema_is_refused(tmp_path: Path) -> None:
    ledger = JobLedger(tmp_path / "jobs")
    job = ledger.create(kind="turn", goal="g")
    step = ledger.begin_step(job.job_id, name="vol-bench", argv=["echo"], idempotent=True)
    path = tmp_path / "jobs" / f"{job.job_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["trace_label"] = "keep-me"
    payload["steps"][0]["trace_step"] = "step-keep"
    path.write_text(json.dumps(payload), encoding="utf-8")
    ledger.finish_step(job.job_id, step.step_id, state=JobState.SUCCEEDED, exit_code=0)
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["trace_label"] == "keep-me"
    assert again["steps"][0]["trace_step"] == "step-keep"
    again["schema_version"] = True
    path.write_text(json.dumps(again), encoding="utf-8")
    with pytest.raises(StoreCorruptError):
        ledger.get(job.job_id)
    assert not path.exists()


def test_reconcile_does_not_overlap_a_transition(tmp_path: Path) -> None:
    ledger = JobLedger(tmp_path / "jobs")
    job = ledger.create(kind="turn", goal="g")
    ledger.transition(job.job_id, JobState.RUNNING)
    inside = {"n": 0, "max": 0}
    guard = threading.Lock()
    real = jobs_mod.atomic_write_json

    def wrapped(path: Path, obj: object, **kwargs: object) -> None:
        with guard:
            inside["n"] += 1
            inside["max"] = max(inside["max"], inside["n"])
        try:
            time.sleep(0.05)
            real(path, obj, **kwargs)  # type: ignore[arg-type]
        finally:
            with guard:
                inside["n"] -= 1

    jobs_mod.atomic_write_json = wrapped
    atomicstore.atomic_write_json = wrapped
    errors: list[BaseException] = []

    def run_transition() -> None:
        try:
            ledger.transition(job.job_id, JobState.SUCCEEDED)
        except Exception as exc:  # noqa: BLE001 — race outcome is either success or illegal
            errors.append(exc)

    def run_reconcile() -> None:
        try:
            ledger.reconcile()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        threads = [threading.Thread(target=run_transition), threading.Thread(target=run_reconcile)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
    finally:
        jobs_mod.atomic_write_json = real
        atomicstore.atomic_write_json = real
    assert inside["max"] == 1
    stored = ledger.get(job.job_id)
    assert stored.state in {JobState.SUCCEEDED, JobState.INTERRUPTED}
    json.loads((tmp_path / "jobs" / f"{job.job_id}.json").read_text(encoding="utf-8"))
    assert errors == [] or all(exc.__class__.__name__ == "IllegalTransition" for exc in errors)


def test_multiprocess_steps_and_single_idempotent_effect(tmp_path: Path) -> None:
    ledger = JobLedger(tmp_path / "jobs")
    job = ledger.create(kind="superpower", goal="g")
    code = (
        "import sys; from pathlib import Path\n"
        "from fx1.interactive.jobs import JobLedger\n"
        "ledger = JobLedger(Path(sys.argv[1])); job, label, n = sys.argv[2], sys.argv[3], int(sys.argv[4])\n"
        "for i in range(n):\n"
        "    ledger.begin_step(job, name='vol-bench', argv=[label, str(i)], "
        "idempotent=True, idempotency_key=f'{label}-{i}')\n"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", code, str(ledger.root), job.job_id, label, "5"])
        for label in ("a", "b", "c", "d")
    ]
    for proc in procs:
        assert proc.wait(timeout=20) == 0
    stored = ledger.get(job.job_id)
    assert len(stored.steps) == 20
    assert len({step.step_id for step in stored.steps}) == 20
    assert stored.revision == 21

    once = (
        "import sys; from pathlib import Path\n"
        "from fx1.interactive.jobs import IllegalTransition, JobLedger\n"
        "ledger = JobLedger(Path(sys.argv[1]))\n"
        "try:\n"
        "    ledger.begin_step(sys.argv[2], name='vol-bench', argv=['echo'], "
        "idempotent=False, idempotency_key='same')\n"
        "    print('started')\n"
        "except IllegalTransition:\n"
        "    print('blocked')\n"
    )
    contested = ledger.create(kind="research", goal="g")
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", once, str(ledger.root), contested.job_id],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    texts = []
    for proc in procs:
        out, _err = proc.communicate(timeout=20)
        assert proc.returncode == 0
        texts.append(out.strip())
    assert sorted(texts) == ["blocked", "started"]
    assert len(ledger.get(contested.job_id).steps) == 1


def test_killed_owner_is_interrupted_and_cancel_persists(tmp_path: Path) -> None:
    code = (
        "import os, sys, time; from pathlib import Path\n"
        "from fx1.interactive.jobs import JobLedger\n"
        "ledger = JobLedger(Path(sys.argv[1]))\n"
        "job = ledger.create(kind='exec', goal='live')\n"
        "idem = sys.argv[2] == '1'\n"
        "step = ledger.begin_step(job.job_id, name='vol-bench', argv=['sleep'], idempotent=idem)\n"
        "ledger.attach_process(job.job_id, step.step_id, pid=os.getpid(), pgid=os.getpid())\n"
        "if sys.argv[3] == 'cancel':\n"
        "    ledger.request_cancel(job.job_id)\n"
        "Path(sys.argv[4]).write_text(job.job_id)\n"
        "time.sleep(30)\n"
    )

    def _spawn(root: Path, *, idempotent: bool, cancel: bool) -> tuple[subprocess.Popen[str], Path]:
        ready = root / "ready"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                code,
                str(root),
                "1" if idempotent else "0",
                "cancel" if cancel else "no",
                str(ready),
            ]
        )
        deadline = time.time() + 10
        while time.time() < deadline and not ready.exists():
            if proc.poll() is not None:
                break
            time.sleep(0.02)
        assert ready.is_file()
        return proc, ready

    root = tmp_path / "idem"
    proc, ready = _spawn(root, idempotent=True, cancel=True)
    job_id = ready.read_text(encoding="utf-8")
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=5)
    ledger = JobLedger(root)
    report = ledger.reconcile()
    assert job_id in report.interrupted
    recovered = ledger.get(job_id)
    assert recovered.state == JobState.INTERRUPTED
    assert recovered.cancel_requested_at
    assert [step.step_id for step in ledger.resumable_steps(job_id)] == [recovered.steps[0].step_id]
    assert len(recovered.steps) == 1

    root = tmp_path / "once"
    proc, ready = _spawn(root, idempotent=False, cancel=False)
    job_id = ready.read_text(encoding="utf-8")
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=5)
    ledger = JobLedger(root)
    ledger.reconcile()
    recovered = ledger.get(job_id)
    assert recovered.state == JobState.INTERRUPTED
    assert ledger.resumable_steps(job_id) == []
    assert len(recovered.steps) == 1


def test_recycled_pid_is_not_treated_as_ours(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    live = proc_root / "7"
    live.mkdir(parents=True)
    (live / "stat").write_text(_stat_text("111"), encoding="utf-8")
    killed: list[int] = []

    def _kill(pid: int, pgid: int | None) -> bool:
        del pgid
        killed.append(pid)
        return True

    ledger = JobLedger(
        tmp_path / "jobs",
        probe=SystemProcessProbe(proc_root=proc_root, platform="linux"),
        killer=_kill,
    )
    job = ledger.create(kind="exec", goal="g")
    step = ledger.begin_step(job.job_id, name="vol-bench", argv=["q"], idempotent=False)
    ledger.attach_process(job.job_id, step.step_id, pid=7, pgid=7)
    (live / "stat").write_text(_stat_text("222"), encoding="utf-8")
    report = ledger.reconcile()
    assert killed == []
    assert report.orphan_running == ()
    assert ledger.get(job.job_id).state == JobState.RUNNING
    (live / "stat").unlink()
    report = ledger.reconcile()
    assert job.job_id in report.interrupted
    assert ledger.get(job.job_id).state == JobState.INTERRUPTED
    assert ledger.resumable_steps(job.job_id) == []


def test_clock_skew_does_not_crash_or_extend_wall(tmp_path: Path) -> None:
    clock = {"now": datetime(2026, 6, 1, tzinfo=UTC)}
    ledger = JobLedger(tmp_path / "jobs", clock=lambda: clock["now"])
    job = ledger.create(kind="turn", goal="g")
    ledger.transition(job.job_id, JobState.RUNNING)
    ledger.transition(job.job_id, JobState.SUCCEEDED)
    clock["now"] = datetime(2020, 1, 1, tzinfo=UTC)
    assert ledger.prune(max_jobs=10, max_age_days=1) == 0
    assert ledger.get(job.job_id).state == JobState.SUCCEEDED

    class _Mono:
        def __init__(self) -> None:
            self.now = 100.0

        def __call__(self) -> float:
            return self.now

    mono = _Mono()
    tracker = UsageTracker(UsageLimits(max_wall_s=10), monotonic=mono)
    tracker.start()
    mono.now = 90.0
    assert tracker.remaining_s() == 10
    assert tracker.deadline() == 100
    tracker.check()
    budget = RunBudget(max_seconds=10, started_at=1_000.0)
    assert budget.expired(now=995.0) is False


def test_budget_race_inf_and_tool_overrides() -> None:
    real = RunBudget.expired

    def slow(self: RunBudget, now: float | None = None) -> bool:
        time.sleep(0.005)
        return real(self, now)

    RunBudget.expired = slow  # type: ignore[method-assign]
    try:
        budget = RunBudget(max_steps=4)
        barrier = threading.Barrier(8)

        def work() -> None:
            barrier.wait(5)
            try:
                budget.charge_step()
            except BudgetError:
                return

        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
        assert budget.steps == 4
    finally:
        RunBudget.expired = real  # type: ignore[method-assign]

    inflated = RunBudget()
    inflated.max_retries = float("inf")
    with pytest.raises(BudgetError, match="real number"):
        inflated.validate()
    with pytest.raises(BudgetError, match="exceeds 32"):
        merge_budget(RunBudget(), {"max_retries": 10**6})
    raised = merge_budget(RunBudget(max_steps=4), {"max_steps": 20})
    assert raised.max_steps == 20
    with pytest.raises(BudgetError, match="cannot be raised"):
        merge_tool_budget(RunBudget(max_steps=4), {"max_steps": 20})
    tightened = merge_tool_budget(
        RunBudget(max_steps=4, max_seconds=30), {"max_steps": 2, "max_seconds": 10}
    )
    assert tightened.max_steps == 2 and tightened.max_seconds == 10
    assert tightened.as_public()["spend_cap"] is None
    assert "not a dollar cap" in tightened.as_public()["spend_cap_note"]


def test_usage_threads_persistence_and_bad_numbers(tmp_path: Path) -> None:
    tracker = UsageTracker(
        UsageLimits(max_steps=10_000, max_model_calls=10_000, max_total_tokens=10_000_000)
    )
    barrier = threading.Barrier(16)

    def work() -> None:
        barrier.wait(5)
        for _ in range(10):
            tracker.charge_step()
            tracker.charge_model_call()
            tracker.charge_usage({"total_tokens": 3})
            tracker.charge_usage(None)

    threads = [threading.Thread(target=work) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    snap = tracker.snapshot()
    assert snap["steps"] == 160
    assert snap["total_tokens"] == 480
    assert snap["unmetered_calls"] == 160
    assert "no spend cap is implied" in tracker.describe()
    path = tmp_path / "budget.json"
    atomic_write_json(path, snap)
    restored = UsageTracker.restore(read_json(path))
    assert restored.snapshot()["total_tokens"] == 480
    with pytest.raises(ValueError):
        UsageLimits.from_dict({"max_wall_s": float("nan")})
    with pytest.raises(ValueError):
        UsageLimits.from_dict({"max_steps": True})
    with pytest.raises(ValueError):
        UsageLimits.from_dict({"max_total_tokens": -1})
    with pytest.raises(ValueError):
        tracker.charge_usage({"total_tokens": float("inf")})  # type: ignore[dict-item]


def test_run_context_ids_reject_traversal() -> None:
    session_id = new_session_id()
    ctx = RunContext(session_id=session_id)
    assert ctx.as_fields()["session_id"] == session_id
    for bad in ("../x", "/tmp/session", "s-20261012T000000Z-\x00abcdef", "é" * 8, ""):
        assert is_valid_id(bad) is False
