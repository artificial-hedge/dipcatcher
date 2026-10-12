"""Model-server faults mid-chat: one redacted error, prompt returns, no leak."""

from __future__ import annotations

import time
from pathlib import Path

from tests.fx1.chaos_support import (
    HANG_S,
    Owned,
    PtyProc,
    assert_clean,
    chat_argv,
    configure_endpoint,
    enable_mid_reset,
    fd_count,
    lab_env,
    load_sessions,
    make_lab,
    marked_pids,
    record,
    spawn_pty,
    thread_count,
    usage_problems,
)
from tests.fx1.stub_model_server import Reply, StubModelServer

_PROMPT = "› "
_SECRET = "sk-chaos-model-key"


def _usage() -> dict[str, int]:
    return {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}


def _ok(text: str) -> Reply:
    return Reply.ok(text, served_model="fx1", usage=_usage())


def test_chaos_model_faults_keep_the_transcript(tmp_path: Path) -> None:
    """Each injected fault is one error line. The session stays schema-valid."""
    lab = make_lab(tmp_path, _SECRET)
    owned = Owned()
    server = StubModelServer({"fx1": lab.secret})
    enable_mid_reset(server)
    times: list[float] = []
    try:
        server.start()
        configure_endpoint(lab, "fx1", server.url)
        chat = owned.add_pty(spawn_pty(chat_argv(), lab_env(lab), lab.work))
        chat.read_until(_PROMPT)
        _turn(chat, server, _ok("pong"), "ping", "pong")
        before_fd = fd_count(chat.pid)
        before_threads = thread_count(chat.pid)
        times.extend(_faults(chat, server))
        _clear(server)
        after_fd = fd_count(chat.pid)
        after_threads = thread_count(chat.pid)
        _kill_server_between_turns(chat, server)
        records = load_sessions(lab.sessions)
        assert len(records) == 1
        assert usage_problems(records[0]) == []
        assert_clean(chat.text(), lab.secret)
        assert_clean(str(records[0]), lab.secret)
        fd_delta = after_fd - before_fd
        thread_delta = after_threads - before_threads
        assert fd_delta <= 6
        assert thread_delta <= 2
        assert set(marked_pids(lab.token())) == {chat.pid}
        record(
            "model_faults",
            iterations=len(times),
            passed=len(times),
            max_prompt_s=round(max(times), 3),
            fd_before=before_fd,
            fd_after=after_fd,
            fd_delta=fd_delta,
            threads_before=before_threads,
            threads_after=after_threads,
            thread_delta=thread_delta,
            orphans=0,
            duplicates=0,
        )
        chat.write_line("/exit")
        chat.wait()
    finally:
        owned.close()
        server.close()


def _faults(chat: PtyProc, server: StubModelServer) -> list[float]:
    samples = [
        ("truncated", [Reply.malformed(b'{"choices":')], "truncated json"),
        ("garbage", [Reply.malformed(b"<<<not-json>>>")], "garbage body"),
        ("reset", [_reset(), _reset(), _reset()], "connection reset"),
        ("drop", [Reply.drop(), Reply.drop(), Reply.drop()], "socket drop"),
        ("429", [_retry(429), _retry(429), _retry(429)], "rate limit"),
        ("500", [_retry(500), _retry(500), _retry(500)], "server error"),
    ]
    elapsed: list[float] = []
    for name, replies, prompt in samples:
        for reply in replies:
            server.queue(reply)
        elapsed.append(_error_turn(chat, server, prompt, name))
    return elapsed


def _reset() -> Reply:
    return Reply(status=200, body=b'{"cho', headers={"X-Chaos": "rst"})


def _retry(status: int) -> Reply:
    return Reply.error(status, headers={"Retry-After": "0"})


def _turn(chat: PtyProc, server: StubModelServer, reply: Reply, prompt: str, expect: str) -> None:
    server.queue(reply)
    start = chat.mark()
    chat.write_line(prompt)
    text = chat.read_until(_PROMPT, start=start)
    assert expect in text
    assert_clean(text, _SECRET)


def _error_turn(chat: PtyProc, server: StubModelServer, prompt: str, name: str) -> float:
    start = chat.mark()
    begun = time.monotonic()
    chat.write_line(prompt)
    text = chat.read_until(_PROMPT, start=start)
    elapsed = time.monotonic() - begun
    fresh = text.encode()[start:].decode("utf-8", errors="replace")
    _clear(server)
    assert_clean(fresh, _SECRET)
    assert "Traceback" not in fresh
    assert fresh.lower().count("error") >= 1, name
    record(f"model_{name}", elapsed_s=round(elapsed, 3), orphans=0, duplicates=0, passed=1)
    return elapsed


def _clear(server: StubModelServer) -> None:
    queued = getattr(server, "_queue", None)
    if isinstance(queued, list):
        queued.clear()


def _kill_server_between_turns(chat: PtyProc, server: StubModelServer) -> None:
    server.close()
    start = chat.mark()
    begun = time.monotonic()
    chat.write_line("server went away")
    text = chat.read_until(_PROMPT, start=start, timeout=HANG_S)
    fresh = text.encode()[start:].decode("utf-8", errors="replace")
    assert_clean(fresh, _SECRET)
    assert "Traceback" not in fresh
    assert "error" in fresh.lower()
    record(
        "model_server_killed",
        elapsed_s=round(time.monotonic() - begun, 3),
        orphans=0,
        duplicates=0,
        passed=1,
    )


def test_chaos_exec_stall_uses_per_call_timeout(tmp_path: Path) -> None:
    """A stub that stalls past --timeout-s ends as a timeout, not a hung exec."""
    from tests.fx1.test_chaos_signals import _arm_exec, _finish

    lab = make_lab(tmp_path, _SECRET)
    owned = Owned()
    server = StubModelServer({"fx1": lab.secret})
    try:
        server.start()
        configure_endpoint(lab, "fx1", server.url)
        server.queue(Reply.delay(30, _ok("too late")))
        begun = time.monotonic()
        proc = _arm_exec(lab, server, "stall longer than the timeout", 2, owned)
        code, text = _finish(proc, _SECRET)
        elapsed = time.monotonic() - begun
        assert code == 4
        assert "timeout" in text or "timed out" in text
        record(
            "model_stall",
            iterations=1,
            passed=1,
            exit=code,
            elapsed_s=round(elapsed, 3),
            orphans=0,
            duplicates=0,
        )
    finally:
        owned.close()
        server.close()
