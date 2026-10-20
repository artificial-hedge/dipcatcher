"""Adversarial checks for cancel, signals, shared config, and process hygiene.

Each test is hermetic, bounded, and kills any child it starts.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from fx1.interactive.headless import ExecRequest, _claim_stop, _mark_done, _RunState, run_exec
from fx1.interactive.model_client import ErrorKind, ModelClient, ModelError, RetryPolicy
from fx1.interactive.procs import ProcessRegistry, ProcLimits, run_owned
from fx1.interactive.terminal import ConsoleWriter, TermCaps
from fx1.serve.backends import ToolCompletion
from tests.fx1.stub_model_server import Reply, StubModelServer

ROOT = Path("/workspace")
PYTHON = ROOT / ".venv" / "bin" / "python"
DIP = ROOT / ".venv" / "bin" / "dipcatcher"
KEY = "rt4-key"


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg"
    sessions = tmp_path / "sessions"
    cfg.mkdir()
    sessions.mkdir()
    monkeypatch.setenv("FX1_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("FX1_SESSION_DIR", str(sessions))
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    for name in (
        "MOONSHOT_API_KEY",
        "FX1_BASE_URL",
        "MOONSHOT_API_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _say_bucket() -> tuple[list[str], Any]:
    lines: list[str] = []

    def say(*args: object, **_kwargs: object) -> None:
        lines.append("" if not args else str(args[0]))

    return lines, say


def _tool(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _completion(content: str, tools: tuple[dict[str, Any], ...] | None = None) -> ToolCompletion:
    return ToolCompletion(
        content=content,
        tool_calls=tools,
        finish_reason="tool_calls" if tools else "stop",
    )


def _host(factory: Any, **kwargs: Any) -> Any:
    from fx1.interactive.concierge import Concierge

    _lines, say = _say_bucket()
    host = Concierge(backend_factory=factory, output_fn=say, animated=False, **kwargs)
    host._lines = _lines  # type: ignore[attr-defined]
    return host


def _endpoint(url: str, model: str = "fx1") -> None:
    from fx1.interactive.profiles import set_endpoint

    set_endpoint(model, KEY, url)


def _fd_count() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill(pid: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def _ppid(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    rest = text.rpartition(")")[2].split()
    if len(rest) < 2 or not rest[1].isdigit():
        return None
    return int(rest[1])


def _descendants(root: int) -> list[int]:
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == root or _ppid(pid) != root:
            continue
        found.append(pid)
        found.extend(_descendants(pid))
    return found


def _child_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["FX1_CONFIG_DIR"] = str(root / "cfg")
    env["FX1_SESSION_DIR"] = str(root / "sessions")
    env["PYTHONPATH"] = str(ROOT / "src")
    env["NO_COLOR"] = "1"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    for name in (
        "MOONSHOT_API_KEY",
        "FX1_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ):
        env.pop(name, None)
    return env


def _popen(args: list[str], env: dict[str, str], cwd: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _stop_proc(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()
    proc.wait(timeout=2)


def _wait_dead(pids: list[int], timeout_s: float = 2.0) -> list[int]:
    deadline = time.monotonic() + timeout_s
    pending = list(pids)
    while pending and time.monotonic() < deadline:
        pending = [pid for pid in pending if _alive(pid)]
        if pending:
            threading.Event().wait(0.02)
    return [pid for pid in pids if _alive(pid)]


def _script(tmp: Path, name: str, body: str) -> Path:
    path = tmp / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Chat-turn cancel points
# ---------------------------------------------------------------------------


def test_cancel_before_backend_open_drops_the_turn(isolated: Path) -> None:
    _endpoint("http://127.0.0.1:9/v1")
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    class Backend:
        supports_cancel = True

        def complete_with_tools(
            self, _messages: list[dict[str, Any]], **_kwargs: Any
        ) -> ToolCompletion:
            calls.append(1)
            return _completion("ANSWER")

    def factory(**_kwargs: Any) -> Backend:
        entered.set()
        release.wait(2)
        return Backend()

    host = _host(factory)
    worker = threading.Thread(target=lambda: host.handle_line("hello"))
    worker.start()
    try:
        assert entered.wait(2)
        host.cancel_turn()
        host.cancel_turn()
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert host.last_status == "cancelled"
        assert host.last_answer == ""
        assert calls == []
        assert host.messages() == []
    finally:
        release.set()
        worker.join(1)


def test_early_cancel_does_not_poison_the_next_turn(isolated: Path) -> None:
    _endpoint("http://127.0.0.1:9/v1")
    entered = threading.Event()
    release = threading.Event()
    seen: list[int] = []

    class Backend:
        supports_cancel = True

        def complete_with_tools(
            self, _messages: list[dict[str, Any]], **_kwargs: Any
        ) -> ToolCompletion:
            seen.append(1)
            return _completion("pinball loss is 0.2")

    def factory(**_kwargs: Any) -> Backend:
        if not entered.is_set():
            entered.set()
            release.wait(2)
        return Backend()

    host = _host(factory)
    first = threading.Thread(target=lambda: host.handle_line("one"))
    first.start()
    try:
        assert entered.wait(2)
        host.cancel_turn()
        release.set()
        first.join(2)
        assert host.last_status == "cancelled"
        host.handle_line("two")
        assert host.last_status == "ok"
        assert host.last_answer == "pinball loss is 0.2"
        assert seen == [1]
    finally:
        release.set()
        first.join(1)


def test_cancel_mid_body_discards_the_answer(isolated: Path) -> None:
    with StubModelServer({"fx1": KEY}) as srv:
        _endpoint(srv.url)
        srv.queue(Reply.delay(5, Reply.ok("SHOULD_NOT_LEAK", served_model="fx1")))
        host = _host(
            lambda **kwargs: ModelClient(retry=RetryPolicy(max_attempts=1), timeout_s=4, **kwargs)
        )
        worker = threading.Thread(target=lambda: host.handle_line("hello"))
        worker.start()
        try:
            assert srv.wait_for_requests(1, timeout=2)
            host.cancel_turn()
            worker.join(2)
            assert not worker.is_alive()
            assert host.last_status == "cancelled"
            assert "SHOULD_NOT_LEAK" not in host.last_answer
            blob = "\n".join(host._lines)
            assert "SHOULD_NOT_LEAK" not in blob
        finally:
            host.cancel_turn()
            worker.join(1)


def test_cancel_during_retry_sleep_skips_the_next_attempt(isolated: Path) -> None:
    with StubModelServer({"fx1": KEY}) as srv:
        _endpoint(srv.url)
        srv.queue(Reply.error(503))
        srv.queue(Reply.ok("retried", served_model="fx1"))

        def factory(**kwargs: Any) -> ModelClient:
            return ModelClient(
                retry=RetryPolicy(max_attempts=3, base_delay_s=0.4, jitter=0),
                timeout_s=3,
                **kwargs,
            )

        host = _host(factory)
        worker = threading.Thread(target=lambda: host.handle_line("hello"))
        worker.start()
        try:
            assert srv.wait_for_requests(1, timeout=2)
            host.cancel_turn()
            worker.join(2)
            assert host.last_status == "cancelled"
            assert len(srv.requests) == 1
            assert "retried" not in host.last_answer
        finally:
            host.cancel_turn()
            worker.join(1)


def test_cancel_during_a_tool_does_not_start_the_next_tool(isolated: Path) -> None:
    _endpoint("http://127.0.0.1:9/v1")
    entered = threading.Event()
    release = threading.Event()
    names: list[str] = []

    class Harness:
        def run(
            self, name: str, _argv: list[str], cancel_event: Any = None, env_extra: Any = None
        ) -> Any:
            del cancel_event, env_extra
            names.append(name)
            if len(names) == 1:
                entered.set()
                release.wait(2)
            return type("R", (), {"stdout": "ok", "stderr": "", "exit_code": 0})()

    class Backend:
        supports_cancel = True
        calls = 0

        def complete_with_tools(
            self, _messages: list[dict[str, Any]], **_kwargs: Any
        ) -> ToolCompletion:
            self.calls += 1
            if self.calls == 1:
                return _completion(
                    "",
                    (
                        _tool("c1", "harness_run", {"name": "doctor", "args": []}),
                        _tool("c2", "harness_run", {"name": "doctor", "args": []}),
                    ),
                )
            return _completion("pinball loss is 0.2")

    backend = Backend()
    host = _host(lambda **_kwargs: backend, harness=Harness())
    worker = threading.Thread(target=lambda: host.handle_line("run doctor twice"))
    worker.start()
    try:
        assert entered.wait(2)
        host.cancel_turn()
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert names == ["doctor"]
        assert backend.calls == 1
        assert host.last_status == "cancelled"
    finally:
        release.set()
        host.cancel_turn()
        worker.join(1)


def test_web_lookup_sees_the_turn_cancel_event(isolated: Path) -> None:
    _endpoint("http://127.0.0.1:9/v1")
    from fx1.webresearch.engine import ResearchReport

    class Researcher:
        def __init__(self) -> None:
            self.event: threading.Event | None = None
            self.entered = threading.Event()

        def run(
            self,
            goal: str,
            budget: Any = None,
            queries: Any = None,
            cancel_event: threading.Event | None = None,
            on_progress: Any = None,
        ) -> ResearchReport:
            del budget, queries, on_progress
            self.event = cancel_event
            self.entered.set()
            if cancel_event is not None:
                cancel_event.wait(1.0)
            return ResearchReport(goal=goal)

    researcher = Researcher()

    class Backend:
        supports_cancel = True
        calls = 0

        def complete_with_tools(
            self, _messages: list[dict[str, Any]], **_kwargs: Any
        ) -> ToolCompletion:
            self.calls += 1
            if self.calls == 1:
                return _completion("", (_tool("w1", "web_lookup", {"query": "pinball loss"}),))
            return _completion("pinball loss is 0.2")

    backend = Backend()
    host = _host(lambda **_kwargs: backend, researcher=researcher)
    worker = threading.Thread(target=lambda: host.handle_line("look up pinball"))
    worker.start()
    try:
        assert researcher.entered.wait(2)
        assert isinstance(researcher.event, threading.Event)
        host.cancel_turn()
        worker.join(2)
        assert researcher.event.is_set()
        assert backend.calls == 1
        assert host.last_status == "cancelled"
    finally:
        host.cancel_turn()
        worker.join(1)


def test_cancel_during_synthesis_withholds_the_draft(isolated: Path) -> None:
    _endpoint("http://127.0.0.1:9/v1")
    entered = threading.Event()

    class Backend:
        supports_cancel = True

        def complete(
            self, _messages: list[dict[str, Any]], cancel_event: Any = None, **_kwargs: Any
        ) -> str:
            entered.set()
            if cancel_event is not None:
                cancel_event.wait(0.8)
            return "SHOULD_NOT_LEAK pinball loss is 0.2"

        def complete_with_tools(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
            return self.complete(messages, kwargs.get("cancel_event"))

    host = _host(lambda **_kwargs: Backend())
    worker = threading.Thread(target=lambda: host.handle_line("/superpower measure pinball --sync"))
    worker.start()
    try:
        assert entered.wait(2)
        started = time.monotonic()
        host.cancel_turn()
        worker.join(2)
        elapsed = time.monotonic() - started
        assert not worker.is_alive()
        assert elapsed < 0.6
        assert host.last_status == "cancelled"
        blob = "\n".join(host._lines) + host.last_answer
        assert "SHOULD_NOT_LEAK" not in blob
    finally:
        host.cancel_turn()
        worker.join(1)


def test_double_cancel_and_cancel_after_shutdown_are_safe(isolated: Path) -> None:
    _endpoint("http://127.0.0.1:9/v1")
    release = threading.Event()
    entered = threading.Event()

    class Backend:
        supports_cancel = True

        def complete_with_tools(
            self, _messages: list[dict[str, Any]], **kwargs: Any
        ) -> ToolCompletion:
            entered.set()
            event = kwargs.get("cancel_event")
            if isinstance(event, threading.Event):
                event.wait(1.0)
                if event.is_set():
                    raise ModelError(
                        ErrorKind.CANCELLED,
                        status=None,
                        retryable=False,
                        retry_after_s=None,
                        model="fx1",
                        host="local",
                        hint="stopped",
                        summary="cancelled",
                    )
            return _completion("pinball loss is 0.2")

    def factory(**_kwargs: Any) -> Backend:
        return Backend()

    host = _host(factory)
    worker = threading.Thread(target=lambda: host.handle_line("hello"))
    worker.start()
    try:
        assert entered.wait(2)
        host.cancel_turn()
        host.cancel_turn()
        worker.join(2)
        assert host.last_status == "cancelled"
        assert host.shutdown(timeout_s=0.5) is True
        host.cancel_turn()
        assert host.cancel_job() is False
        assert host.handle_line("/cancel") is False
    finally:
        release.set()
        host.cancel_turn()
        worker.join(1)


def test_cancel_job_storm_stops_the_worker_once(isolated: Path) -> None:
    from fx1.interactive.concierge import BackgroundJob, Concierge

    stopped = threading.Event()

    def run_fn(cancel_event: threading.Event, _on_event: Any) -> str:
        cancel_event.wait(2)
        stopped.set()
        return "done"

    host = Concierge(output_fn=lambda *_a, **_k: None, animated=False)
    host._job = BackgroundJob(label="research", run_fn=run_fn).start()
    errors: list[BaseException] = []

    def storm() -> None:
        try:
            for _ in range(20):
                host.cancel_job()
        except BaseException as exc:  # noqa: BLE001 — the storm must not raise
            errors.append(exc)

    threads = [threading.Thread(target=storm) for _ in range(4)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        assert host._job.thread is not None
        host._job.thread.join(2)
        assert stopped.is_set()
        assert not host._job.thread.is_alive()
        assert errors == []
        assert host.cancel_job() is False
    finally:
        host.cancel_job()
        if host._job.thread is not None:
            host._job.thread.join(1)


def test_shutdown_joins_a_job_that_watches_cancel(isolated: Path) -> None:
    from fx1.interactive.concierge import BackgroundJob, Concierge

    def run_fn(cancel_event: threading.Event, _on_event: Any) -> str:
        cancel_event.wait(3)
        return "stopped"

    host = Concierge(output_fn=lambda *_a, **_k: None, animated=False)
    job = BackgroundJob(label="research", run_fn=run_fn).start()
    host._job = job
    try:
        assert host.shutdown(timeout_s=1) is True
        assert job.thread is not None and not job.thread.is_alive()
        assert job.cancel_event.is_set()
    finally:
        job.cancel()
        if job.thread is not None:
            job.thread.join(1)


def test_exec_timeout_cancels_the_background_job(isolated: Path) -> None:
    from fx1.interactive.concierge import Concierge
    from fx1.webresearch.engine import ResearchReport

    class Researcher:
        def __init__(self) -> None:
            self.event: threading.Event | None = None

        def run(
            self,
            goal: str,
            budget: Any = None,
            queries: Any = None,
            cancel_event: threading.Event | None = None,
            on_progress: Any = None,
        ) -> ResearchReport:
            del budget, queries, on_progress
            self.event = cancel_event
            if cancel_event is not None:
                cancel_event.wait(3)
            return ResearchReport(goal=goal)

    researcher = Researcher()
    _endpoint("http://127.0.0.1:9/v1")

    def builder(model: str, **kwargs: Any) -> Concierge:
        kwargs["researcher"] = researcher
        return Concierge(model, **kwargs)

    started = time.monotonic()
    outcome = run_exec(
        ExecRequest(prompt="/research pinball loss", timeout_s=0.4, model="fx1"),
        concierge_factory=builder,
        install_signals=False,
    )
    elapsed = time.monotonic() - started
    assert outcome.status == "timeout"
    assert elapsed < 2.0
    assert isinstance(researcher.event, threading.Event)
    assert researcher.event.is_set()
    names = [
        thread.name
        for thread in threading.enumerate()
        if thread.is_alive() and thread.name.startswith("dip-")
    ]
    assert names == []


# ---------------------------------------------------------------------------
# Model client races
# ---------------------------------------------------------------------------


def test_decoded_completion_survives_a_deadline_that_arrives_afterward(isolated: Path) -> None:
    del isolated
    flipped = {"on": False}

    def clock() -> float:
        return 100.0 if flipped["on"] else 0.0

    original = ModelClient._success

    def success(self: ModelClient, *args: Any, **kwargs: Any) -> Any:
        outcome = original(self, *args, **kwargs)
        flipped["on"] = True
        return outcome

    ModelClient._success = success  # type: ignore[method-assign]
    try:
        with StubModelServer({"fx1": KEY}) as srv:
            srv.queue(Reply.ok("kept-answer", served_model="fx1", usage={"total_tokens": 3}))
            client = ModelClient(
                KEY,
                "fx1",
                srv.url,
                timeout_s=5,
                retry=RetryPolicy(max_attempts=1),
                monotonic=clock,
            )
            assert (
                client.complete([{"role": "user", "content": "hi"}], deadline=1.0) == "kept-answer"
            )
            assert client.total_usage.get("total_tokens") == 3
    finally:
        ModelClient._success = original  # type: ignore[method-assign]


def test_decoded_completion_survives_a_cancel_that_arrives_afterward(isolated: Path) -> None:
    del isolated
    cancel = threading.Event()
    original = ModelClient._success

    def success(self: ModelClient, *args: Any, **kwargs: Any) -> Any:
        outcome = original(self, *args, **kwargs)
        cancel.set()
        return outcome

    ModelClient._success = success  # type: ignore[method-assign]
    try:
        with StubModelServer({"fx1": KEY}) as srv:
            srv.queue(Reply.ok("kept-answer", served_model="fx1"))
            client = ModelClient(
                KEY, "fx1", srv.url, timeout_s=5, retry=RetryPolicy(max_attempts=1)
            )
            text = client.complete([{"role": "user", "content": "hi"}], cancel_event=cancel)
            assert text == "kept-answer"
    finally:
        ModelClient._success = original  # type: ignore[method-assign]


def test_concierge_cancel_during_connect_leaves_no_http_thread(isolated: Path) -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(0)
    port = server.getsockname()[1]
    filler = socket.socket()
    filler.settimeout(1)
    filler.connect(("127.0.0.1", port))
    _endpoint(f"http://127.0.0.1:{port}/v1")

    def factory(**kwargs: Any) -> ModelClient:
        return ModelClient(retry=RetryPolicy(max_attempts=1), timeout_s=3, **kwargs)

    host = _host(factory)
    worker = threading.Thread(target=lambda: host.handle_line("hello"))
    worker.start()
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if any(
                thread.name == "fx1-model-http" and thread.is_alive()
                for thread in threading.enumerate()
            ):
                break
            threading.Event().wait(0.02)
        host.cancel_turn()
        worker.join(2)
        assert not worker.is_alive()
        assert host.last_status == "cancelled"
        leaked = [
            thread.name
            for thread in threading.enumerate()
            if thread.is_alive() and thread.name == "fx1-model-http"
        ]
        assert leaked == []
    finally:
        host.cancel_turn()
        worker.join(1)
        filler.close()
        server.close()


def test_two_hundred_cancelled_connects_drop_threads_and_fds(isolated: Path) -> None:
    del isolated
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(0)
    port = server.getsockname()[1]
    filler = socket.socket()
    filler.settimeout(1)
    filler.connect(("127.0.0.1", port))
    before = _fd_count()
    cancel = threading.Event()
    errors: list[str] = []
    threads: list[threading.Thread] = []

    def one() -> None:
        client = ModelClient(
            KEY,
            "fx1",
            f"http://127.0.0.1:{port}/v1/chat/completions",
            timeout_s=1.5,
            retry=RetryPolicy(max_attempts=1),
        )
        try:
            client.complete([{"role": "user", "content": "x"}], cancel_event=cancel)
        except ModelError:
            return
        except Exception as exc:  # noqa: BLE001 — the assertion reports the type
            errors.append(type(exc).__name__)

    try:
        for _ in range(200):
            thread = threading.Thread(target=one, daemon=True)
            threads.append(thread)
            thread.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            live = [
                item
                for item in threading.enumerate()
                if item.name == "fx1-model-http" and item.is_alive()
            ]
            if len(live) >= 20:
                break
            threading.Event().wait(0.02)
        cancel.set()
        for thread in threads:
            thread.join(1.5)
        leaked = [
            item.name
            for item in threading.enumerate()
            if item.name == "fx1-model-http" and item.is_alive()
        ]
        after = _fd_count()
        assert errors == []
        assert leaked == []
        assert all(not thread.is_alive() for thread in threads)
        assert after - before < 40, f"fd delta {after - before}"
    finally:
        cancel.set()
        filler.close()
        server.close()
        for thread in threads:
            thread.join(0.2)


# ---------------------------------------------------------------------------
# exec signals and the deadline thread
# ---------------------------------------------------------------------------


def test_finished_turn_wins_the_deadline_claim(isolated: Path) -> None:
    del isolated
    state = _RunState(
        request=ExecRequest(prompt="x", timeout_s=1),
        deadline=1.0,
        monotonic=time.monotonic,
        registry=object(),
        install_signals=False,
        concierge_factory=None,
        backend_factory=None,
        session_store=None,
    )
    assert _claim_stop(state) is True
    assert state.timed_out.is_set()
    state.timed_out.clear()
    _mark_done(state)
    assert _claim_stop(state) is False
    assert not state.timed_out.is_set()


def test_exec_joins_its_deadline_thread(isolated: Path) -> None:
    _endpoint("http://127.0.0.1:9/v1")

    class Backend:
        supports_cancel = True

        def complete_with_tools(
            self, _messages: list[dict[str, Any]], **_kwargs: Any
        ) -> ToolCompletion:
            return _completion("pinball loss is 0.1")

    outcome = run_exec(
        ExecRequest(prompt="hello", timeout_s=5, model="fx1"),
        backend_factory=lambda **_kwargs: Backend(),
        install_signals=False,
    )
    assert outcome.status == "ok"
    assert outcome.exit_code == 0
    live = [
        thread.name
        for thread in threading.enumerate()
        if thread.is_alive() and thread.name == "fx1-exec-deadline"
    ]
    assert live == []
    foreign = [
        thread.name
        for thread in threading.enumerate()
        if thread.is_alive()
        and not thread.daemon
        and thread is not threading.main_thread()
        and not thread.name.startswith("pytest")
    ]
    assert foreign == []


def test_sigterm_while_the_registry_lock_is_held_exits_143(isolated: Path) -> None:
    script = _script(
        isolated,
        "lock_signal.py",
        """
import os, signal, sys, time
sys.path.insert(0, "/workspace/src")
from fx1.interactive.headless import ExecRequest, run_exec
from fx1.interactive.procs import ProcessRegistry
from fx1.interactive.profiles import set_endpoint
from fx1.serve.backends import ToolCompletion
set_endpoint("fx1", "k", "http://127.0.0.1:9/v1")
registry = ProcessRegistry()

class Backend:
    supports_cancel = True
    def complete_with_tools(self, messages, **kwargs):
        with registry._lock:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(2)
        return ToolCompletion(content="x", tool_calls=None, finish_reason="stop")

outcome = run_exec(
    ExecRequest(prompt="hello", timeout_s=4, model="fx1"),
    backend_factory=lambda **kwargs: Backend(),
    registry=registry,
    install_signals=True,
)
raise SystemExit(outcome.exit_code)
""",
    )
    proc = _popen([str(PYTHON), str(script)], _child_env(isolated), isolated)
    try:
        out, err = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        _stop_proc(proc)
        pytest.fail("exec hung while the registry lock was held")
    finally:
        _stop_proc(proc)
    assert proc.returncode == 143, f"out={out!r} err={err!r}"


def test_sigint_before_the_model_factory_returns_exits_130(isolated: Path) -> None:
    ready = isolated / "ready"
    script = _script(
        isolated,
        "before_request.py",
        f"""
import sys, time
from pathlib import Path
sys.path.insert(0, "/workspace/src")
from fx1.interactive.headless import ExecRequest, run_exec
from fx1.interactive.profiles import set_endpoint
from fx1.serve.backends import ToolCompletion
set_endpoint("fx1", "k", "http://127.0.0.1:9/v1")

def factory(**kwargs):
    Path({str(ready)!r}).write_text("ready", encoding="utf-8")
    time.sleep(8)
    class Backend:
        supports_cancel = True
        def complete_with_tools(self, messages, **kw):
            return ToolCompletion(content="x", tool_calls=None, finish_reason="stop")
    return Backend()

outcome = run_exec(
    ExecRequest(prompt="hello", timeout_s=10, model="fx1"),
    backend_factory=factory,
    install_signals=True,
)
raise SystemExit(outcome.exit_code)
""",
    )
    proc = _popen([str(PYTHON), str(script)], _child_env(isolated), isolated)
    try:
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            threading.Event().wait(0.02)
        assert ready.exists()
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=3)
        assert proc.returncode == 130, f"out={out!r} err={err!r}"
        assert _wait_dead(_descendants(proc.pid)) == []
    finally:
        _stop_proc(proc)


def _exec_signal_during_body(isolated: Path, signum: int, code: int) -> None:
    with StubModelServer({"fx1": KEY}) as srv:
        _endpoint(srv.url)
        srv.queue(Reply.delay(8, Reply.ok("SHOULD_NOT_LEAK", served_model="fx1")))
        proc = _popen(
            [str(DIP), "exec", "--timeout-s", "15", "--model", "fx1", "hello"],
            _child_env(isolated),
            isolated,
        )
        try:
            assert srv.wait_for_requests(1, timeout=10)
            kids = _descendants(proc.pid)
            proc.send_signal(signum)
            out, err = proc.communicate(timeout=3)
            assert proc.returncode == code, f"out={out!r} err={err!r}"
            assert "SHOULD_NOT_LEAK" not in out
            assert _wait_dead(kids) == []
        finally:
            _stop_proc(proc)


@pytest.mark.parametrize(
    ("signum", "code"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_exec_signal_during_the_response_body(isolated: Path, signum: int, code: int) -> None:
    _exec_signal_during_body(isolated, signum, code)


def test_sigterm_during_a_blocked_connect_exits_143(isolated: Path) -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(0)
    port = server.getsockname()[1]
    filler = socket.socket()
    filler.connect(("127.0.0.1", port))
    ready = isolated / "connect-ready"
    script = _script(
        isolated,
        "connect_block.py",
        f"""
import sys
from pathlib import Path
sys.path.insert(0, "/workspace/src")
from fx1.interactive.headless import ExecRequest, run_exec
from fx1.interactive.model_client import ModelClient, RetryPolicy
from fx1.interactive.profiles import set_endpoint
set_endpoint("fx1", "k", "http://127.0.0.1:{port}/v1")

def factory(**kwargs):
    Path({str(ready)!r}).write_text("ready", encoding="utf-8")
    kwargs["retry"] = RetryPolicy(max_attempts=1)
    return ModelClient(**kwargs)

outcome = run_exec(
    ExecRequest(prompt="hello", timeout_s=8, model="fx1"),
    backend_factory=factory,
    install_signals=True,
)
raise SystemExit(outcome.exit_code)
""",
    )
    proc = _popen([str(PYTHON), str(script)], _child_env(isolated), isolated)
    try:
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            threading.Event().wait(0.02)
        assert ready.exists()
        threading.Event().wait(0.1)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=3)
        assert proc.returncode == 143, f"out={out!r} err={err!r}"
    finally:
        _stop_proc(proc)
        filler.close()
        server.close()


_STUBBORN = """
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
grandchild = os.fork()
if grandchild == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)
    os._exit(0)
Path(sys.argv[1]).write_text(f"{os.getpid()}\\n{grandchild}\\n", encoding="utf-8")
time.sleep(30)
"""


def _signal_kills_stubborn_tree(isolated: Path, signum: int, code: int) -> None:
    stubborn = _script(isolated, "stubborn.py", _STUBBORN)
    pidfile = isolated / "pids"
    driver = _script(
        isolated,
        "tree_exec.py",
        f"""
import sys
sys.path.insert(0, "/workspace/src")
from fx1.interactive.headless import ExecRequest, run_exec
from fx1.interactive.procs import ProcLimits, run_owned
from fx1.interactive.profiles import set_endpoint
from fx1.serve.backends import ToolCompletion
set_endpoint("fx1", "k", "http://127.0.0.1:9/v1")
SCRIPT = {str(stubborn)!r}
PIDFILE = {str(pidfile)!r}
PY = {str(PYTHON)!r}

class Harness:
    def run(self, name, argv, cancel_event=None, env_extra=None):
        return run_owned(
            [PY, SCRIPT, PIDFILE],
            ProcLimits(timeout_s=20, kill_grace_s=0.4),
            cancel_event=cancel_event,
        )

class Backend:
    supports_cancel = True
    def complete_with_tools(self, messages, **kwargs):
        return ToolCompletion(
            content="",
            tool_calls=({{
                "id": "h1",
                "type": "function",
                "function": {{"name": "harness_run", "arguments": '{{"name": "doctor", "args": []}}'}},
            }},),
            finish_reason="tool_calls",
        )

outcome = run_exec(
    ExecRequest(prompt="run it", timeout_s=15, model="fx1"),
    backend_factory=lambda **kwargs: Backend(),
    harness=Harness(),
    install_signals=True,
)
raise SystemExit(outcome.exit_code)
""",
    )
    proc = _popen([str(PYTHON), str(driver)], _child_env(isolated), isolated)
    pids: list[int] = []
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not pidfile.exists():
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=1)
                pytest.fail(f"driver exited early {proc.returncode} out={out!r} err={err!r}")
            threading.Event().wait(0.02)
        assert pidfile.exists()
        pids = [int(line) for line in pidfile.read_text(encoding="utf-8").split() if line.strip()]
        assert len(pids) == 2
        proc.send_signal(signum)
        out, err = proc.communicate(timeout=4)
        assert proc.returncode == code, f"out={out!r} err={err!r}"
        assert _wait_dead(pids) == [], f"orphans {_wait_dead(pids)}"
    finally:
        _stop_proc(proc)
        for pid in pids:
            _kill(pid)


@pytest.mark.parametrize(
    ("signum", "code"),
    [(signal.SIGTERM, 143), (signal.SIGINT, 130)],
)
def test_stubborn_grandchildren_die_with_the_exec_signal(
    isolated: Path, signum: int, code: int
) -> None:
    _signal_kills_stubborn_tree(isolated, signum, code)


def test_cancelled_owned_process_reaps_reader_threads(isolated: Path) -> None:
    registry = ProcessRegistry()
    cancel = threading.Event()
    spawned = threading.Event()

    def emit(event: str, _fields: dict[str, Any]) -> None:
        if event == "proc.spawn":
            spawned.set()

    def run() -> None:
        run_owned(
            [str(PYTHON), "-c", "import time; time.sleep(30)"],
            ProcLimits(timeout_s=5, kill_grace_s=0.3),
            cancel_event=cancel,
            registry=registry,
            emit=emit,
        )

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert spawned.wait(2)
        cancel.set()
        worker.join(2)
        assert not worker.is_alive()
        readers = [
            thread.name
            for thread in threading.enumerate()
            if thread.is_alive() and thread.name.startswith("proc-")
        ]
        assert readers == []
        assert registry.live() == []
    finally:
        cancel.set()
        registry.terminate_all(0.2)
        worker.join(1)


# ---------------------------------------------------------------------------
# Shared config and in-process thread safety
# ---------------------------------------------------------------------------


def test_two_processes_keep_credentials_sessions_jobs_and_history(isolated: Path) -> None:
    script = _script(
        isolated,
        "share.py",
        """
import sys
sys.path.insert(0, "/workspace/src")
from pathlib import Path
from fx1.interactive.jobs import JobLedger
from fx1.interactive.profiles import config_dir, set_endpoint
from fx1.interactive.sessions import SessionStore
from fx1.interactive.terminal import ConsoleWriter, LineReader, TermCaps
role = sys.argv[1]
model = "fx1" if role == "a" else "fx1-lite"
set_endpoint(model, "key-" + role, "https://example.test/v1")
SessionStore().create(model=model, messages=[{"role": "user", "content": "from-" + role}])
JobLedger().create(kind="chat", goal="goal-" + role)
caps = TermCaps(False, False, False, False, True, 80, 24, False)
writer = ConsoleWriter(caps)
reader = LineReader(caps, writer, history_path=config_dir() / "history", input_fn=lambda _prompt: "hist-" + role)
reader.read_message("dip> ")
reader.save_history()
""",
    )
    env = _child_env(isolated)
    procs = [_popen([str(PYTHON), str(script), role], env, isolated) for role in ("a", "b")]
    try:
        for proc in procs:
            out, err = proc.communicate(timeout=4)
            assert proc.returncode == 0, f"out={out!r} err={err!r}"
    finally:
        for proc in procs:
            _stop_proc(proc)
    from fx1.interactive.jobs import JobLedger
    from fx1.interactive.profiles import config_dir, resolve_endpoint

    fx1 = resolve_endpoint("fx1")
    lite = resolve_endpoint("fx1-lite")
    assert fx1 is not None and fx1[0] == "key-a"
    assert lite is not None and lite[0] == "key-b"
    goals = {record.goal for record in JobLedger().list(limit=10)}
    assert goals == {"goal-a", "goal-b"}
    bodies: list[str] = []
    for path in (isolated / "sessions").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for message in payload.get("messages", []):
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                bodies.append(content)
    assert "from-a" in bodies and "from-b" in bodies
    history = (config_dir() / "history").read_text(encoding="utf-8")
    assert "hist-a" in history and "hist-b" in history


def test_last_fields_stay_readable_while_a_worker_writes(isolated: Path) -> None:
    from fx1.interactive.concierge import Concierge
    from fx1.interactive.model_ux import CallSnapshot

    host = Concierge(output_fn=lambda *_a, **_k: None, animated=False)
    stop = threading.Event()
    errors: list[str] = []

    def writer() -> None:
        index = 0
        while not stop.is_set():
            host._remember_answer(f"pinball loss is 0.{index % 10}")
            host._append_message({"role": "assistant", "content": f"n-{index}"})
            host._note_call(
                CallSnapshot(usage={"total_tokens": index % 5 + 1}, identity=None, latency_s=0.01)
            )
            index += 1

    def reader() -> None:
        while not stop.is_set():
            try:
                _ = host.last_answer
                _ = host.last_status
                _ = host.last_usage
                _ = host.last_identity
                _ = host.last_latency_s
                _ = host.messages()
                _ = host.active_model
            except Exception as exc:  # noqa: BLE001 — report any torn read
                errors.append(type(exc).__name__)
                stop.set()

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for thread in threads:
        thread.start()
    threading.Event().wait(0.25)
    stop.set()
    for thread in threads:
        thread.join(1)
    assert errors == []
    assert isinstance(host.last_answer, str)
    assert isinstance(host.messages(), list)


def test_console_writer_keeps_lines_whole_across_threads(isolated: Path) -> None:
    import io

    caps = TermCaps(False, False, False, False, True, 80, 24, False)
    stream = io.StringIO()
    writer = ConsoleWriter(caps, out=stream)
    errors: list[str] = []

    def emit(index: int) -> None:
        try:
            for n in range(30):
                writer.line(f"T{index}-{n}-PINBALL")
        except Exception as exc:  # noqa: BLE001
            errors.append(type(exc).__name__)

    threads = [threading.Thread(target=emit, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    text = stream.getvalue()
    assert errors == []
    for index in range(6):
        for n in range(30):
            assert f"T{index}-{n}-PINBALL" in text


def test_concurrent_job_drain_does_not_raise(isolated: Path) -> None:
    from fx1.interactive.concierge import BackgroundJob

    del isolated
    job = BackgroundJob(label="x", run_fn=lambda _cancel, _event: None)
    errors: list[str] = []

    def produce() -> None:
        for index in range(3000):
            job.events.append(("step", str(index)))

    def consume() -> None:
        try:
            for _ in range(3000):
                job.drain()
        except Exception as exc:  # noqa: BLE001
            errors.append(type(exc).__name__)

    threads = [threading.Thread(target=produce)]
    threads.extend(threading.Thread(target=consume) for _ in range(3))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert errors == []


def test_tick_while_a_job_emits_does_not_crash(isolated: Path) -> None:
    from fx1.interactive.concierge import BackgroundJob, Concierge

    def run_fn(cancel_event: threading.Event, on_event: Any) -> str:
        for index in range(100):
            if cancel_event.is_set():
                return "cancelled"
            on_event("step", f"n-{index}")
        return "done"

    host = Concierge(output_fn=lambda *_a, **_k: None, animated=False)
    job = BackgroundJob(label="research", run_fn=run_fn).start()
    host._job = job
    try:
        while job.running:
            host.tick()
        host.tick()
        assert job.thread is not None and not job.thread.is_alive()
    finally:
        job.cancel()
        if job.thread is not None:
            job.thread.join(1)
