#!/usr/bin/env python3
"""Launch a Cursor cloud agent to run CI and fail closed on its report.

The Actions job is only a launcher. Heavy checks run in the agent started
by ``POST /v0/agents``. This process polls ``GET /v0/agents/{id}`` and passes
only when that agent finished and the commit status context ``cursor-ci``
(plus the single PR comment, when there is a PR) reports pass.

Authentication is HTTP Basic with the API key as the username and an empty
password, matching the v0 Cloud Agents API. The key is read from the
environment and is never written to logs, payloads, or comments.

Missing configuration, HTTP errors, timeouts, and unparseable reports exit
non-zero. A parsed CI failure also exits non-zero.
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

API_ROOT = "https://api.cursor.com"
GITHUB_API = "https://api.github.com"
STATUS_CONTEXT = "cursor-ci"
COMMENT_MARKER = "<!-- cursor-ci -->"
RESULT_RE = re.compile(r"CURSOR_CI_RESULT:\s*(pass|fail)\b")
DEFAULT_PROMPT = Path(".github/cursor-ci/prompt.md")
DEFAULT_TIMEOUT_S = 90 * 60
DEFAULT_POLL_S = 30.0
REQUEST_TIMEOUT_S = 60.0
TERMINAL_STATUSES = frozenset(
    {"FINISHED", "ERROR", "EXPIRED", "CANCELLED", "CANCELED", "FAILED", "ARCHIVED"}
)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_URL_RE = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_AGENT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_EVENT_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


class CursorCIError(Exception):
    """Fail-closed CI launcher error. The message is safe to print."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


class HttpClient(Protocol):
    """Structural request callable. Tests pass a fake; production uses urllib."""

    def __call__(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class Settings:
    api_key: str
    github_token: str
    sha: str
    repository_url: str
    head_repo: str
    base_repo: str
    pr_number: str | None
    ref_name: str
    event_name: str
    timeout_s: float
    poll_s: float
    prompt_path: Path


@dataclass(frozen=True)
class Verdict:
    passed: bool
    kind: str
    reason: str


def redact(text: str, secrets: list[str]) -> str:
    """Remove API keys and their Basic-auth encodings from text."""
    redacted = text
    for secret in secrets:
        if len(secret) < 6:
            continue
        redacted = redacted.replace(secret, "[REDACTED]")
        encoded = base64.b64encode(f"{secret}:".encode()).decode("ascii")
        redacted = redacted.replace(encoded, "[REDACTED]")
    return redacted


def cursor_headers(api_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{api_key}:".encode()).decode("ascii")
    return {
        "Accept": "application/json",
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "User-Agent": "dipcatcher-cursor-ci",
    }


def github_headers(token: str, *, json_body: bool) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "dipcatcher-cursor-ci",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def normalize_repository_url(url: str) -> str:
    text = url.strip()
    if text.endswith(".git"):
        text = text[:-4]
    if text.endswith("/"):
        text = text[:-1]
    if not _URL_RE.fullmatch(text):
        raise CursorCIError("repository URL is not an https GitHub URL without credentials")
    return text


def head_repo_from_url(url: str) -> str:
    return normalize_repository_url(url).removeprefix("https://github.com/")


def build_launch_body(*, repository: str, ref: str, prompt: str) -> dict[str, object]:
    """v0 launch body. ``source.ref`` is the full head SHA.

    ``source.prUrl`` is omitted on purpose: the v0 API ignores repository and
    ref when a pull request URL is set. ``autoCreatePr`` stays false so the
    agent is not asked to open a pull request.
    """
    if not _SHA_RE.fullmatch(ref):
        raise CursorCIError("head sha must be 40 lowercase hex characters")
    if not prompt.strip():
        raise CursorCIError("CI prompt is empty")
    return {
        "prompt": {"text": prompt},
        "source": {"repository": normalize_repository_url(repository), "ref": ref},
        "target": {"autoCreatePr": False},
    }


def assemble_prompt(template: str, settings: Settings) -> str:
    pr = settings.pr_number or "(none)"
    context = "\n".join(
        (
            "## Run context (authoritative for this launch)",
            "",
            f"- event: {settings.event_name}",
            f"- head repository URL: {settings.repository_url}",
            f"- head repo: {settings.head_repo}",
            f"- base repo: {settings.base_repo}",
            f"- head sha: {settings.sha}",
            f"- head ref name: {settings.ref_name or '(none)'}",
            f"- pull request number: {pr}",
            "",
            "Check out exactly this sha. If `git rev-parse HEAD` differs, report fail.",
            "",
        )
    )
    return template.rstrip() + "\n\n" + context


def agent_id_from_launch(payload: object) -> str:
    if not isinstance(payload, dict):
        raise CursorCIError("launch response is not an object")
    agent_id = payload.get("id")
    if not isinstance(agent_id, str) or not _AGENT_ID_RE.fullmatch(agent_id):
        raise CursorCIError("launch response missing agent id")
    return agent_id


def agent_status(payload: object) -> str:
    if not isinstance(payload, dict):
        raise CursorCIError("agent status response is not an object")
    status = payload.get("status")
    if not isinstance(status, str) or not status.strip():
        raise CursorCIError("agent status response missing status")
    return status.strip().upper()


def agent_page_url(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    target = payload.get("target")
    if not isinstance(target, dict):
        return None
    url = target.get("url")
    if not isinstance(url, str) or "\n" in url or len(url) > 500:
        return None
    if not url.startswith("https://"):
        return None
    return url


def result_token(text: str | None) -> str | None:
    """Return pass/fail only when the marker appears exactly once."""
    if not text:
        return None
    found = RESULT_RE.findall(text)
    if len(found) != 1:
        return None
    token = found[0]
    if isinstance(token, str) and token in {"pass", "fail"}:
        return token
    return None


def select_commit_status(payload: object) -> tuple[str | None, str | None]:
    """Latest combined-status entry whose context is ``cursor-ci``."""
    if not isinstance(payload, dict):
        raise CursorCIError("commit status response is not an object")
    entries = payload.get("statuses")
    if not isinstance(entries, list):
        raise CursorCIError("commit status response missing statuses")
    matches = [
        item for item in entries if isinstance(item, dict) and item.get("context") == STATUS_CONTEXT
    ]
    if not matches:
        return None, None
    matches.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""))
    chosen = matches[-1]
    state = chosen.get("state")
    description = chosen.get("description")
    if not isinstance(state, str):
        return None, None
    if description is not None and not isinstance(description, str):
        description = None
    return state, description


def marker_comments(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, list):
        raise CursorCIError("PR comment response is not a list")
    found: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        body = item.get("body")
        if isinstance(body, str) and COMMENT_MARKER in body:
            found.append(item)
    found.sort(key=_comment_id)
    return found


def _comment_id(item: dict[str, object]) -> int:
    value = item.get("id")
    if isinstance(value, int):
        return value
    return 0


def judge(
    *,
    agent_status_value: str,
    commit_state: str | None,
    description: str | None,
    comments: list[dict[str, object]],
    pr_number: str | None,
) -> Verdict:
    """Pass only on FINISHED plus an unambiguous success report."""
    if agent_status_value != "FINISHED":
        return Verdict(False, "unparseable", f"agent status {agent_status_value} is not FINISHED")
    description_token = result_token(description)
    if commit_state == "failure" and description_token == "fail":
        return Verdict(False, "ci-fail", "cursor-ci reported failure")
    if commit_state != "success" or description_token != "pass":
        if commit_state is None:
            return Verdict(False, "unparseable", "missing commit status context cursor-ci")
        return Verdict(False, "unparseable", "unparseable cursor-ci status")
    if pr_number is None:
        return Verdict(True, "pass", "cursor-ci passed")
    if len(comments) != 1:
        return Verdict(
            False,
            "unparseable",
            f"expected one {COMMENT_MARKER} PR comment, found {len(comments)}",
        )
    body = comments[0].get("body")
    if result_token(body if isinstance(body, str) else None) != "pass":
        return Verdict(False, "unparseable", "PR comment is missing CURSOR_CI_RESULT: pass")
    return Verdict(True, "pass", "cursor-ci passed")


def failure_comment(reason: str, *, sha: str, agent_id: str | None) -> str:
    who = agent_id or "(not launched)"
    safe = reason.replace("\r", " ").replace("\n", " ").replace("|", "/")
    return "\n".join(
        (
            COMMENT_MARKER,
            "## cursor-ci",
            "",
            f"head: `{sha}`",
            f"agent: `{who}`",
            "",
            "| job | result | first failing lines |",
            "| --- | --- | --- |",
            f"| cursor-ci-launcher | fail | {safe} |",
            "",
            "CURSOR_CI_RESULT: fail",
            "",
        )
    )


def _positive_float(raw: str, field: str) -> float:
    try:
        number = float(raw)
    except ValueError as exc:
        raise CursorCIError(f"{field} is not a number") from exc
    if number <= 0 or number != number or number > 24 * 60 * 60:
        raise CursorCIError(f"{field} is out of range")
    return number


def _one_line(
    value: str, field: str, *, max_len: int, pattern: re.Pattern[str] | None = None
) -> str:
    if not value or len(value) > max_len or any(ord(ch) < 32 for ch in value):
        raise CursorCIError(f"{field} is empty or not a single line")
    if pattern is not None and not pattern.fullmatch(value):
        raise CursorCIError(f"{field} has an unexpected shape")
    return value


def _optional_line(value: str, field: str, *, max_len: int) -> str:
    if any(ord(ch) < 32 for ch in value) or len(value) > max_len:
        raise CursorCIError(f"{field} is not a single line")
    return value


def load_settings(environ: Mapping[str, str], *, cwd: Path | None = None) -> Settings:
    missing = [
        name
        for name in ("CURSOR_API_KEY", "GITHUB_TOKEN", "CURSOR_CI_SHA", "CURSOR_CI_REPOSITORY_URL")
        if not environ.get(name, "").strip()
    ]
    if missing:
        raise CursorCIError("missing " + ", ".join(missing))
    api_key = environ["CURSOR_API_KEY"].strip()
    github_token = environ["GITHUB_TOKEN"].strip()
    sha = _one_line(environ["CURSOR_CI_SHA"].strip(), "head sha", max_len=40, pattern=_SHA_RE)
    repository_url = normalize_repository_url(environ["CURSOR_CI_REPOSITORY_URL"])
    derived_head = head_repo_from_url(repository_url)
    head_raw = environ.get("CURSOR_CI_HEAD_REPO", "").strip()
    if head_raw:
        head_repo = _one_line(head_raw, "head repo", max_len=200, pattern=_REPO_RE)
        if head_repo != derived_head:
            raise CursorCIError("head repo does not match the repository URL")
    else:
        head_repo = derived_head
    base_raw = environ.get("CURSOR_CI_BASE_REPO", "").strip() or head_repo
    base_repo = _one_line(base_raw, "base repo", max_len=200, pattern=_REPO_RE)
    pr_raw = environ.get("CURSOR_CI_PR_NUMBER", "").strip()
    if pr_raw in {"", "null", "0"}:
        pr_number = None
    elif pr_raw.isdigit():
        pr_number = str(int(pr_raw))
    else:
        raise CursorCIError("pull request number is not an integer")
    timeout_s = _positive_float(
        environ.get("CURSOR_CI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_S)),
        "timeout",
    )
    poll_s = _positive_float(
        environ.get("CURSOR_CI_POLL_SECONDS", str(int(DEFAULT_POLL_S))),
        "poll interval",
    )
    if poll_s > timeout_s:
        raise CursorCIError("poll interval is longer than the timeout")
    prompt_raw = environ.get("CURSOR_CI_PROMPT", "").strip()
    prompt_path = Path(prompt_raw) if prompt_raw else DEFAULT_PROMPT
    if not prompt_path.is_absolute():
        prompt_path = (cwd or Path.cwd()) / prompt_path
    if not prompt_path.is_file():
        raise CursorCIError(f"CI prompt file is missing: {prompt_path.name}")
    event_name = _one_line(
        environ.get("CURSOR_CI_EVENT", "workflow_dispatch").strip() or "workflow_dispatch",
        "event",
        max_len=64,
        pattern=_EVENT_RE,
    )
    ref_name = _optional_line(
        environ.get("CURSOR_CI_REF_NAME", "").strip(), "ref name", max_len=256
    )
    return Settings(
        api_key=api_key,
        github_token=github_token,
        sha=sha,
        repository_url=repository_url,
        head_repo=head_repo,
        base_repo=base_repo,
        pr_number=pr_number,
        ref_name=ref_name,
        event_name=event_name,
        timeout_s=timeout_s,
        poll_s=poll_s,
        prompt_path=prompt_path,
    )


def path_only(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return parsed.path or "/"


def urllib_request(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> HttpResponse:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            return HttpResponse(int(response.status), payload)
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        return HttpResponse(int(exc.code), payload)
    except urllib.error.URLError as exc:
        raise CursorCIError(f"request to {path_only(url)} failed") from exc


def request_json(
    http: HttpClient,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    body: bytes | None,
    sleep: Callable[[float], None],
    retries: int,
    secrets: list[str],
) -> object:
    last_status = 0
    last_body = b""
    for attempt in range(retries + 1):
        try:
            response = http(method, url, headers=headers, body=body, timeout=REQUEST_TIMEOUT_S)
        except CursorCIError:
            raise
        except AssertionError:
            raise
        except Exception as exc:
            raise CursorCIError(f"request to {path_only(url)} failed") from exc
        last_status = response.status
        last_body = response.body
        if 200 <= response.status < 300:
            if not response.body:
                return {}
            try:
                loaded = json.loads(response.body.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise CursorCIError(f"unparseable response from {method} {path_only(url)}") from exc
            return loaded
        if response.status in RETRY_STATUSES and attempt < retries:
            sleep(float(2**attempt))
            continue
        snippet = redact(last_body.decode("utf-8", errors="replace")[:300], secrets)
        raise CursorCIError(f"HTTP {last_status} from {method} {path_only(url)}: {snippet}")
    raise CursorCIError(f"HTTP {last_status} from {method} {path_only(url)}")


def launch_agent(
    http: HttpClient,
    settings: Settings,
    prompt: str,
    *,
    sleep: Callable[[float], None],
) -> tuple[str, str | None]:
    body = build_launch_body(
        repository=settings.repository_url,
        ref=settings.sha,
        prompt=prompt,
    )
    payload = request_json(
        http,
        "POST",
        f"{API_ROOT}/v0/agents",
        headers=cursor_headers(settings.api_key),
        body=json.dumps(body).encode("utf-8"),
        sleep=sleep,
        retries=0,
        secrets=[settings.api_key, settings.github_token],
    )
    return agent_id_from_launch(payload), agent_page_url(payload)


def poll_agent(
    http: HttpClient,
    settings: Settings,
    agent_id: str,
    *,
    sleep: Callable[[float], None],
    now: Callable[[], float],
) -> dict[str, object]:
    deadline = now() + settings.timeout_s
    url = f"{API_ROOT}/v0/agents/{urllib.parse.quote(agent_id)}"
    headers = cursor_headers(settings.api_key)
    secrets = [settings.api_key, settings.github_token]
    while True:
        payload = request_json(
            http,
            "GET",
            url,
            headers=headers,
            body=None,
            sleep=sleep,
            retries=3,
            secrets=secrets,
        )
        status = agent_status(payload)
        print(f"cursor-ci: agent {agent_id} status {status}", flush=True)
        if not isinstance(payload, dict):
            raise CursorCIError("agent status response is not an object")
        if status in TERMINAL_STATUSES:
            return payload
        if now() >= deadline:
            _stop_agent(http, settings, agent_id, sleep=sleep)
            raise CursorCIError(
                f"timed out after {int(settings.timeout_s)}s waiting for agent {agent_id}; "
                f"last status {status}"
            )
        remaining = deadline - now()
        sleep(min(settings.poll_s, remaining))


def _stop_agent(
    http: HttpClient,
    settings: Settings,
    agent_id: str,
    *,
    sleep: Callable[[float], None],
) -> None:
    url = f"{API_ROOT}/v0/agents/{urllib.parse.quote(agent_id)}/stop"
    try:
        request_json(
            http,
            "POST",
            url,
            headers=cursor_headers(settings.api_key),
            body=b"{}",
            sleep=sleep,
            retries=0,
            secrets=[settings.api_key, settings.github_token],
        )
    except CursorCIError as exc:
        print(f"cursor-ci: stop request failed: {exc}", file=sys.stderr, flush=True)


def fetch_commit_status(
    http: HttpClient,
    settings: Settings,
    *,
    sleep: Callable[[float], None],
) -> tuple[str | None, str | None]:
    url = f"{GITHUB_API}/repos/{settings.head_repo}/commits/{settings.sha}/status"
    payload = request_json(
        http,
        "GET",
        url,
        headers=github_headers(settings.github_token, json_body=False),
        body=None,
        sleep=sleep,
        retries=3,
        secrets=[settings.api_key, settings.github_token],
    )
    return select_commit_status(payload)


def fetch_comments(
    http: HttpClient,
    settings: Settings,
    *,
    sleep: Callable[[float], None],
) -> list[dict[str, object]]:
    if settings.pr_number is None:
        return []
    url = (
        f"{GITHUB_API}/repos/{settings.base_repo}/issues/{settings.pr_number}/comments?per_page=100"
    )
    payload = request_json(
        http,
        "GET",
        url,
        headers=github_headers(settings.github_token, json_body=False),
        body=None,
        sleep=sleep,
        retries=3,
        secrets=[settings.api_key, settings.github_token],
    )
    return marker_comments(payload)


def upsert_failure_comment(
    http: HttpClient,
    settings: Settings,
    *,
    reason: str,
    agent_id: str | None,
    sleep: Callable[[float], None],
) -> None:
    if settings.pr_number is None:
        return
    existing = fetch_comments(http, settings, sleep=sleep)
    if len(existing) == 1:
        body = existing[0].get("body")
        if result_token(body if isinstance(body, str) else None) is not None:
            return
    text = failure_comment(reason, sha=settings.sha, agent_id=agent_id)
    encoded = json.dumps({"body": text}).encode("utf-8")
    headers = github_headers(settings.github_token, json_body=True)
    secrets = [settings.api_key, settings.github_token]
    if len(existing) == 1 and isinstance(existing[0].get("id"), int):
        comment_id = existing[0]["id"]
        url = f"{GITHUB_API}/repos/{settings.base_repo}/issues/comments/{comment_id}"
        request_json(
            http,
            "PATCH",
            url,
            headers=headers,
            body=encoded,
            sleep=sleep,
            retries=0,
            secrets=secrets,
        )
        return
    if len(existing) > 1:
        return
    url = f"{GITHUB_API}/repos/{settings.base_repo}/issues/{settings.pr_number}/comments"
    request_json(
        http, "POST", url, headers=headers, body=encoded, sleep=sleep, retries=0, secrets=secrets
    )


def _install_stop_handler(
    http: HttpClient,
    settings: Settings,
    agent_id: str,
    sleep: Callable[[float], None],
) -> None:
    def handle(signum: int, frame: object) -> None:
        del signum, frame
        _stop_agent(http, settings, agent_id, sleep=sleep)
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def _execute(
    settings: Settings,
    http: HttpClient,
    *,
    sleep: Callable[[float], None],
    now: Callable[[], float],
    handle_signals: bool,
) -> int:
    template = settings.prompt_path.read_text(encoding="utf-8")
    prompt = assemble_prompt(template, settings)
    agent_id, page = launch_agent(http, settings, prompt, sleep=sleep)
    print(f"cursor-ci: launched {agent_id}", flush=True)
    if page is not None:
        print(f"cursor-ci: agent url {page}", flush=True)
    if handle_signals:
        _install_stop_handler(http, settings, agent_id, sleep)
    final = poll_agent(http, settings, agent_id, sleep=sleep, now=now)
    state, description = fetch_commit_status(http, settings, sleep=sleep)
    comments = fetch_comments(http, settings, sleep=sleep)
    verdict = judge(
        agent_status_value=agent_status(final),
        commit_state=state,
        description=description,
        comments=comments,
        pr_number=settings.pr_number,
    )
    if not verdict.passed:
        upsert_failure_comment(
            http,
            settings,
            reason=verdict.reason,
            agent_id=agent_id,
            sleep=sleep,
        )
        print(f"cursor-ci: fail: {verdict.reason}", file=sys.stderr)
        return 1
    print(f"cursor-ci: pass: {verdict.reason}")
    return 0


def run(
    environ: Mapping[str, str] | None = None,
    *,
    http: HttpClient | None = None,
    sleep: Callable[[float], None] | None = None,
    now: Callable[[], float] | None = None,
    cwd: Path | None = None,
    handle_signals: bool | None = None,
) -> int:
    env = os.environ if environ is None else environ
    client = urllib_request if http is None else http
    sleeper = time.sleep if sleep is None else sleep
    clock = time.monotonic if now is None else now
    signals = http is None if handle_signals is None else handle_signals
    holder: dict[str, Settings] = {}
    try:
        settings = load_settings(env, cwd=cwd)
        holder["settings"] = settings
        return _execute(settings, client, sleep=sleeper, now=clock, handle_signals=signals)
    except CursorCIError as exc:
        failed = holder.get("settings")
        if failed is not None and failed.pr_number is not None:
            try:
                upsert_failure_comment(
                    client,
                    failed,
                    reason=str(exc),
                    agent_id=None,
                    sleep=sleeper,
                )
            except CursorCIError as comment_exc:
                print(
                    f"cursor-ci: also failed to comment: {comment_exc}",
                    file=sys.stderr,
                )
        print(f"cursor-ci: fail: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        print("cursor-ci: fail: this script takes no arguments", file=sys.stderr)
        return 1
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
