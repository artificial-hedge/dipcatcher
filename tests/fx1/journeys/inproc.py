"""Drive a real Concierge in-process. The runner only submits user lines."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stderr
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.fx1.journeys.checks import final_problem
from tests.fx1.journeys.textutil import substitute


class _UrlInput:
    """Supplies the endpoint URL when ``/keys set`` prompts for it."""

    def __init__(self) -> None:
        self.url = ""

    def __call__(self, prompt: str = "") -> str:
        del prompt
        if not self.url:
            raise EOFError
        return self.url


class _Err:
    """Append stderr chunks in order with stdout lines."""

    def __init__(self, parts: list[str]) -> None:
        self.parts = parts

    def write(self, data: str | bytes) -> int:
        if isinstance(data, bytes):
            self.parts.append(data.decode("utf-8", errors="replace"))
            return len(data)
        self.parts.append(data)
        return len(data)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


class _Transcript:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.err = _Err(self.parts)

    def add(self, text: str) -> None:
        self.parts.append(str(text) + "\n")

    def text(self) -> str:
        return "".join(self.parts)


def run_inproc(
    journey: dict[str, Any],
    paths: dict[str, Path],
    *,
    stub: Any,
    stub_url: str,
) -> tuple[str | None, str, int | None, bool]:
    """Return ``(problem, transcript, exit_code, ended)``."""
    from fx1.interactive.approvals import ApprovalGate
    from fx1.interactive.concierge import Concierge
    from fx1.interactive.sessions import SessionStore

    transcript = _Transcript()
    url_input = _UrlInput()
    store = SessionStore(paths["sessions"])
    concierge = Concierge(
        output_fn=transcript.add,
        animated=False,
        input_fn=url_input,
        session_store=store,
        approvals=ApprovalGate(),
    )
    try:
        problem, ended = _drive(concierge, journey, transcript, stub, stub_url, paths, url_input)
    finally:
        _stop(concierge)
    if problem:
        return problem, transcript.text(), None, ended
    final = final_problem(
        journey.get("final", {}),
        paths=paths,
        transcript=transcript.text(),
        exit_code=None,
        ended=ended,
        stub_url=stub_url,
    )
    return final, transcript.text(), None, ended


def _drive(
    concierge: Any,
    journey: dict[str, Any],
    transcript: _Transcript,
    stub: Any,
    stub_url: str,
    paths: dict[str, Path],
    url_input: _UrlInput,
) -> tuple[str | None, bool]:
    ended = False
    steps = journey["steps"]
    for index, step in enumerate(steps):
        problem, ended_now = _one_step(
            concierge, step, transcript, stub, stub_url, str(paths["work"]), url_input
        )
        ended = ended or ended_now
        if problem:
            return problem, ended
        if ended and index != len(steps) - 1:
            return "session ended before the remaining steps", ended
    return None, ended


def _one_step(
    concierge: Any,
    step: dict[str, Any],
    transcript: _Transcript,
    stub: Any,
    stub_url: str,
    work: str,
    url_input: _UrlInput,
) -> tuple[str | None, bool]:
    from tests.fx1.journeys.match import match_output, wait_requests

    before = transcript.text()
    seen = 0 if stub is None else len(stub.requests)
    say = _opt(step.get("say"), stub_url, work)
    secret = _opt(step.get("secret"), stub_url, work)
    url_input.url = _opt(step.get("url"), stub_url, work) or ""
    with redirect_stderr(transcript.err):
        ended = _submit(concierge, say, secret)
        concierge.tick()
        if step.get("wait") == "job_done":
            concierge.join_job(timeout_s=_hang(step))
            concierge.tick()
    if step.get("wait") == "request" and not wait_requests(None, stub, seen, _hang(step)):
        return "model request was not observed", ended
    if step.get("wait") == "job_done" and concierge.job_running:
        return "background job still running after join", ended
    region = transcript.text()[len(before) :]
    return match_output(step, region, stub_url=stub_url, work=work), ended


def _submit(concierge: Any, say: str | None, secret: str | None) -> bool:
    if not say:
        return False
    with _secret_patch(secret):
        return bool(concierge.handle_line(say))


def _secret_patch(secret: str | None) -> Any:
    if secret is None:
        return nullcontext()

    def fake(prompt: str = "", stream: object = None) -> str:
        del prompt, stream
        return secret

    return patch("getpass.getpass", fake)


def _opt(value: object, stub_url: str, work: str) -> str | None:
    if not isinstance(value, str):
        return None
    return substitute(value, stub_url=stub_url, work=work)


def _hang(step: dict[str, Any]) -> float:
    from tests.fx1.journeys.match import hang_seconds

    return hang_seconds(step)


def _stop(concierge: Any) -> None:
    if not concierge.job_running:
        return
    concierge.cancel_job()
    concierge.join_job(timeout_s=5)
