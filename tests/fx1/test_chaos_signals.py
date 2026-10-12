"""SIGTERM, SIGHUP, and SIGINT against chat and dipcatcher exec."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.fx1.chaos_support import (
    HANG_S,
    Owned,
    PtyProc,
    assert_clean,
    chat_argv,
    configure_endpoint,
    fd_count,
    kill_pids,
    lab_env,
    lane_plan,
    load_sessions,
    make_lab,
    marked_pids,
    marker_count,
    marker_pids,
    pid_alive,
    record,
    spawn_exec,
    spawn_pty,
    wait_until,
)
from tests.fx1.stub_model_server import Reply, StubModelServer

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM")
_PROMPT = "› "
_SECRET = "sk-chaos-signal-key"


def _usage() -> dict[str, int]:
    return {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def _exec_argv(prompt: str, timeout_s: float) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dipcatcher_cli",
        "exec",
        "--model",
        "fx1",
        "--json",
        "--timeout-s",
        str(timeout_s),
        prompt,
    ]


def _arm_exec(
    lab: object, server: StubModelServer, prompt: str, timeout_s: float, owned: Owned
) -> subprocess.Popen[str]:
    from tests.fx1.chaos_support import Lab

    assert isinstance(lab, Lab)
    proc = spawn_exec(_exec_argv(prompt, timeout_s), lab_env(lab), lab.work)
    owned.add_pid(proc.pid or 0)
    assert server.wait_for_requests(len(server.requests) + 1, timeout=HANG_S)
    return proc


def _finish(proc: subprocess.Popen[str], secret: str) -> tuple[int, str]:
    try:
        out, err = proc.communicate(timeout=HANG_S)
    except subprocess.TimeoutExpired:
        if proc.pid:
            os_kill(proc.pid)
        out, err = proc.communicate(timeout=5)
    text = (out or "") + (err or "")
    assert_clean(text, secret)
    assert "Traceback" not in text
    return proc.returncode if proc.returncode is not None else -1, text


def os_kill(pid: int) -> None:
    import os

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def test_chaos_exec_sigint_sigterm_sighup_and_timeout(tmp_path: Path) -> None:
    """Exec uses documented 130/143, treats SIGHUP as interrupt, and honors --timeout-s."""
    lab = make_lab(tmp_path, _SECRET)
    owned = Owned()
    server = StubModelServer({"fx1": lab.secret})
    results: list[tuple[str, int, float]] = []
    try:
        server.start()
        configure_endpoint(lab, "fx1", server.url)
        results.append(_signal_exec(lab, server, owned, signal.SIGINT, 130, "sigint"))
        results.append(_signal_exec(lab, server, owned, signal.SIGTERM, 143, "sigterm"))
        if hasattr(signal, "SIGHUP"):
            results.append(_signal_exec(lab, server, owned, signal.SIGHUP, 130, "sighup"))
        results.append(_timeout_exec(lab, server, owned))
        record(
            "exec_signals",
            iterations=len(results),
            passed=len(results),
            orphans=len(marked_pids(lab.token())),
            duplicates=0,
            max_exit_s=max(item[2] for item in results),
            codes=[item[1] for item in results],
        )
        assert marked_pids(lab.token()) == []
    finally:
        kill_pids(list(owned.pids))
        owned.close()
        server.close()


def _signal_exec(
    lab: object,
    server: StubModelServer,
    owned: Owned,
    sig: int,
    expected: int,
    name: str,
) -> tuple[str, int, float]:
    server.queue(Reply.delay(30, Reply.ok("late", served_model="fx1", usage=_usage())))
    started = time.monotonic()
    proc = _arm_exec(lab, server, f"signal {name}", 30, owned)
    assert proc.pid is not None
    proc.send_signal(sig)
    code, _text = _finish(proc, _SECRET)
    elapsed = time.monotonic() - started
    assert code == expected
    record(f"exec_{name}", exit=code, elapsed_s=round(elapsed, 3), orphans=0, duplicates=0)
    return name, code, elapsed


def _timeout_exec(lab: object, server: StubModelServer, owned: Owned) -> tuple[str, int, float]:
    server.queue(Reply.delay(30, Reply.ok("late", served_model="fx1", usage=_usage())))
    started = time.monotonic()
    proc = _arm_exec(lab, server, "stall past the deadline", 2, owned)
    code, text = _finish(proc, _SECRET)
    elapsed = time.monotonic() - started
    assert code == 4
    assert "timeout" in text or "timed out" in text
    record("exec_timeout", exit=code, elapsed_s=round(elapsed, 3), orphans=0, duplicates=0)
    return "timeout", code, elapsed


def test_chaos_chat_sigterm_reaps_the_lab_child(tmp_path: Path) -> None:
    """SIGTERM during /superpower leaves no lane-power child and a readable session."""
    _chat_signal(tmp_path, signal.SIGTERM, "chat_sigterm")


def test_chaos_chat_sighup_reaps_the_lab_child(tmp_path: Path) -> None:
    """SIGHUP (terminal closed) during /superpower leaves no lane-power child."""
    if not hasattr(signal, "SIGHUP"):
        pytest.skip("SIGHUP")
    _chat_signal(tmp_path, signal.SIGHUP, "chat_sighup")


def test_chaos_chat_sigint_storm_stops_the_job(tmp_path: Path) -> None:
    """A SIGINT storm cancels the background job, then leaves the chat."""
    lab = make_lab(tmp_path, _SECRET)
    owned = Owned()
    server = StubModelServer({"fx1": lab.secret})
    started = time.monotonic()
    try:
        chat = _start_lane(lab, owned, server)
        before = marker_count(lab.marker)
        mark = chat.mark()
        chat.signal(signal.SIGINT)
        text = chat.read_until("cancel requested", start=mark)
        assert_clean(text, lab.secret)
        assert "Traceback" not in text
        wait_until(
            lambda: not any(pid_alive(pid) for pid in marker_pids(lab.marker)),
            HANG_S,
            "lane-power survived cancel",
        )
        code = _sigint_exit(chat)
        assert "Traceback" not in chat.text()
        assert marker_count(lab.marker) == before
        records = load_sessions(lab.sessions)
        assert records and records[0]["schema"] == 1
        assert marked_pids(lab.token()) == []
        record(
            "chat_sigint_storm",
            iterations=1,
            passed=1,
            exit=code,
            elapsed_s=round(time.monotonic() - started, 3),
            orphans=0,
            duplicates=0,
            session_job=records[0]["job"]["state"],
        )
    finally:
        kill_pids(marker_pids(lab.marker))
        kill_pids(marked_pids(lab.token()))
        owned.close()
        server.close()


def _chat_signal(tmp_path: Path, sig: int, scenario: str) -> None:
    lab = make_lab(tmp_path, _SECRET)
    owned = Owned()
    server = StubModelServer({"fx1": lab.secret})
    started = time.monotonic()
    try:
        chat = _start_lane(lab, owned, server)
        before = marker_count(lab.marker)
        fds = fd_count(chat.pid)
        chat.signal(sig)
        code = chat.wait()
        wait_until(
            lambda: not any(pid_alive(pid) for pid in marker_pids(lab.marker)),
            HANG_S,
            "lane-power survived the chat signal",
        )
        assert "Traceback" not in chat.text()
        assert_clean(chat.text(), lab.secret)
        records = load_sessions(lab.sessions)
        assert len(records) == 1
        if sig == signal.SIGTERM:
            assert code == 143
            assert records[0]["job"]["state"] != "running"
        assert marker_count(lab.marker) == before
        assert marked_pids(lab.token()) == []
        assert descendants_clear(chat.pid)
        record(
            scenario,
            iterations=1,
            passed=1,
            exit=code,
            elapsed_s=round(time.monotonic() - started, 3),
            orphans=0,
            duplicates=0,
            session_job=records[0]["job"]["state"],
            fds_at_signal=fds,
        )
    finally:
        kill_pids(marker_pids(lab.marker))
        kill_pids(marked_pids(lab.token()))
        owned.close()
        server.close()


def _sigint_exit(chat: PtyProc) -> int:
    """Second Ctrl-C inside the console's 2s window exits after the job stops."""
    deadline = time.monotonic() + HANG_S
    while time.monotonic() < deadline:
        if chat.proc.poll() is not None:
            return chat.wait(timeout=5)
        start = chat.mark()
        chat.signal(signal.SIGINT)
        token = _next_token(chat, start, min(deadline, time.monotonic() + 10))
        if token == "press Ctrl-C again":
            chat.signal(signal.SIGINT)
            return chat.wait()
    raise AssertionError(f"chat did not exit on SIGINT tail={chat.text()[-400:]}")


def _next_token(chat: PtyProc, start: int, deadline: float) -> str:
    tokens = ("press Ctrl-C again", "cancel requested")
    while time.monotonic() < deadline:
        fresh = chat.buf[start:]
        for token in tokens:
            if token.encode() in fresh:
                return token
        if chat.proc.poll() is not None:
            return ""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        _read_some(chat, remaining)
    return ""


def _read_some(chat: PtyProc, remaining: float) -> None:
    import select

    ready, _, _ = select.select([chat.master], [], [], min(0.2, remaining))
    if not ready:
        return
    try:
        chunk = os.read(chat.master, 4096)
    except OSError:
        return
    if chunk:
        chat.buf.extend(chunk)


def descendants_clear(pid: int) -> bool:
    from tests.fx1.chaos_support import descendants

    return descendants(pid) == []


def _start_lane(lab: object, owned: Owned, server: StubModelServer) -> PtyProc:
    from tests.fx1.chaos_support import Lab

    assert isinstance(lab, Lab)
    server.start()
    configure_endpoint(lab, "fx1", server.url)
    out = lab.config / "runs" / "lane"
    out.mkdir(parents=True, exist_ok=True)
    server.queue(Reply.ok(lane_plan(out), served_model="fx1", usage=_usage()))
    chat = owned.add_pty(spawn_pty(chat_argv(), lab_env(lab), lab.work))
    chat.read_until(_PROMPT)
    start = chat.mark()
    chat.write_line("/superpower lane power alarm rate")
    chat.read_until("running in the background", start=start)
    chat.read_until(_PROMPT, start=start)
    wait_until(lambda: marker_count(lab.marker) >= 1, HANG_S, "lane-power did not start")
    return chat
