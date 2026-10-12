"""SIGKILL the real chat during a background /superpower and resume it."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.fx1.chaos_support import (
    HANG_S,
    Owned,
    assert_clean,
    broken_json,
    chat_argv,
    configure_endpoint,
    empty_plan,
    job_payloads,
    kill_pids,
    lab_env,
    lane_plan,
    load_sessions,
    make_lab,
    marked_pids,
    marker_count,
    marker_pids,
    nonterminal_jobs,
    pid_alive,
    record,
    spawn_pty,
    stray_temps,
    wait_until,
)
from tests.fx1.stub_model_server import Reply, StubModelServer

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="SIGKILL")
_PROMPT = "› "
_SECRET = "sk-chaos-not-a-real-key"


def _usage() -> dict[str, int]:
    return {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}


def _run(
    argv: list[str], lab_env_map: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        env=lab_env_map,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=HANG_S,
        check=False,
    )


def test_chaos_sigkill_mid_superpower_then_resume(tmp_path: Path) -> None:
    """A killed chat leaves a valid session, does not rerun the step, and resumes."""
    lab = make_lab(tmp_path, _SECRET)
    owned = Owned()
    server = StubModelServer({"fx1": lab.secret})
    started = time.monotonic()
    exit_code = -1
    try:
        server.start()
        configure_endpoint(lab, "fx1", server.url)
        out = lab.config / "runs" / "lane"
        out.mkdir(parents=True, exist_ok=True)
        server.queue(Reply.ok(lane_plan(out), served_model="fx1", usage=_usage()))
        chat = owned.add_pty(spawn_pty(chat_argv(), lab_env(lab), lab.work))
        chat.read_until(_PROMPT)
        chat.write_line("/superpower lane power alarm rate")
        chat.read_until("running in the background")
        wait_until(lambda: marker_count(lab.marker) >= 1, HANG_S, "lane-power did not start")
        wait_until(lambda: bool(nonterminal_jobs(lab.config)), HANG_S, "job was not non-terminal")
        before = marker_count(lab.marker)
        chat.signal(signal.SIGKILL)
        exit_code = chat.wait()
        assert exit_code == -signal.SIGKILL
        wait_until(
            lambda: not any(pid_alive(pid) for pid in marker_pids(lab.marker)),
            HANG_S,
            "lane-power child survived SIGKILL of the chat",
        )
        _killed_files_are_intact(lab)
        assert marker_count(lab.marker) == before
        _sessions_cli(lab)
        resume_s = _resume(lab, owned, server, before)
        duplicates = marker_count(lab.marker) - before
        orphans = len(marked_pids(lab.token()))
        assert duplicates == 0
        assert orphans == 0
        record(
            "sigkill_resume",
            iterations=1,
            passed=1,
            exit=exit_code,
            resume_s=round(resume_s, 3),
            elapsed_s=round(time.monotonic() - started, 3),
            orphans=orphans,
            duplicates=duplicates,
            temps=len(stray_temps(lab.sessions, lab.config)),
        )
    finally:
        kill_pids(marker_pids(lab.marker))
        kill_pids(marked_pids(lab.token()))
        owned.close()
        server.close()


def _killed_files_are_intact(lab: object) -> None:
    from tests.fx1.chaos_support import Lab

    assert isinstance(lab, Lab)
    records = load_sessions(lab.sessions)
    assert len(records) == 1
    assert records[0]["schema"] == 1
    assert_clean(str(records[0]), lab.secret)
    assert nonterminal_jobs(lab.config)
    assert broken_json(lab.config, lab.sessions, lab.work) == []


def _sessions_cli(lab: object) -> None:
    from tests.fx1.chaos_support import Lab

    assert isinstance(lab, Lab)
    env = lab_env(lab)
    base = [sys.executable, "-m", "dipcatcher_cli", "sessions"]
    listed = _run([*base, "list", "--json"], env, lab.work)
    assert_clean(listed.stdout + listed.stderr, lab.secret)
    assert listed.returncode == 0
    shown = _run([*base, "show", "last"], env, lab.work)
    assert_clean(shown.stdout + shown.stderr, lab.secret)
    assert shown.returncode == 0
    assert "job_state:" in shown.stdout


def _resume(lab: object, owned: Owned, server: StubModelServer, before: int) -> float:
    from tests.fx1.chaos_support import Lab

    assert isinstance(lab, Lab)
    server.queue(Reply.ok(empty_plan(), served_model="fx1", usage=_usage()))
    server.queue(Reply.ok("No capability was selected.", served_model="fx1", usage=_usage()))
    started = time.monotonic()
    chat = owned.add_pty(spawn_pty(chat_argv(resume=True), lab_env(lab), lab.work))
    text = chat.read_until(_PROMPT)
    assert_clean(text, lab.secret)
    assert "Traceback" not in text
    assert "not restarted" in text or "interrupted" in text
    assert marker_count(lab.marker) == before
    resumed = load_sessions(lab.sessions)
    assert resumed[0]["job"]["state"] == "interrupted"
    jobs = job_payloads(lab.config)
    assert jobs
    assert all(job.get("state") == "interrupted" for job in jobs)
    mark = chat.mark()
    chat.write_line("/superpower status check only")
    follow = chat.read_until(_PROMPT, start=mark)
    assert_clean(follow, lab.secret)
    assert "Traceback" not in follow
    assert marker_count(lab.marker) == before
    chat.write_line("/exit")
    chat.wait()
    return time.monotonic() - started
