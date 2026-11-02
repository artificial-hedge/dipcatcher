"""Fail-closed behavior of the Cursor cloud-agent CI launcher."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "ci" / "cursor_ci.py"
_spec = importlib.util.spec_from_file_location("dipcatcher_cursor_ci", _SCRIPT)
assert _spec is not None and _spec.loader is not None
cursor_ci = importlib.util.module_from_spec(_spec)
sys.modules["dipcatcher_cursor_ci"] = cursor_ci
_spec.loader.exec_module(cursor_ci)

SHA = "a" * 40
KEY = "cursor_test_key_not_real_0123456789"
TOKEN = "ghp_test_token_not_real_0123456789"
GATE = (
    "github.event_name != 'pull_request' && github.event_name != 'push' "
    "|| vars.USE_CURSOR_CI == 'false'"
)
PUSH_STILL_RUNS = "github.event_name != 'pull_request' || vars.USE_CURSOR_CI == 'false'"
GATED_WORKFLOWS = (
    "ci.yml",
    "fx1.yml",
    "dependency-audit.yml",
    "full-coverage.yml",
    "matrix.yml",
    "perf-baseline.yml",
    "proofcore.yml",
    "replay_viz.yml",
    "reproduce_sota.yml",
    "secret-scan.yml",
    "simtest.yml",
    "web_explorer.yml",
    "witness_monitor.yml",
    "adversarial-properties.yml",
    "arch_guards.yml",
    "atlas.yml",
    "docs.yml",
)
UNTOUCHED_WORKFLOWS = (
    "codeql.yml",
    "dependency-review.yml",
    "scorecard.yml",
    "release.yml",
    "property-nightly.yml",
)
# Jobs that must keep running on push (or only on workflow_dispatch) and so
# do not use the full pull_request-and-push gate.
SPECIAL_JOBS = {
    ("ci.yml", "package"): PUSH_STILL_RUNS,
    ("docs.yml", "build"): PUSH_STILL_RUNS,
}


def _jobs(text: str) -> dict[str, str]:
    lines = text.splitlines()
    try:
        start = lines.index("jobs:")
    except ValueError:
        return {}
    jobs: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" "):
            break
        if (
            line.startswith("  ")
            and not line.startswith("   ")
            and line.rstrip().endswith(":")
            and not line.strip().startswith("#")
        ):
            if name is not None:
                jobs[name] = "\n".join(buf)
            name = line.strip()[:-1]
            buf = [line]
        elif name is not None:
            buf.append(line)
    if name is not None:
        jobs[name] = "\n".join(buf)
    return jobs


def _job_if(block: str) -> str:
    for line in block.splitlines():
        if line.startswith("    if:"):
            value = line.split(":", 1)[1].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                return value[1:-1]
            return value
    return ""


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class Api:
    def __init__(
        self,
        statuses: list[str],
        *,
        commit_state: str | None = "success",
        description: str | None = "CURSOR_CI_RESULT: pass",
        comments: list[dict[str, Any]] | None = None,
        launch_status: int = 200,
        launch_body: dict[str, Any] | None = None,
    ) -> None:
        self.statuses = list(statuses)
        self.commit_state = commit_state
        self.description = description
        self.comments: list[dict[str, Any]] = list(comments or [])
        self.launch_status = launch_status
        self.launch_body = launch_body or {
            "id": "bc_test",
            "status": "CREATING",
            "target": {"url": "https://cursor.com/agents?id=bc_test"},
        }
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.posted: list[bytes] = []

    def __call__(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> Any:
        del timeout
        assert KEY not in url
        assert TOKEN not in url
        self.calls.append((method, url, dict(headers), body))
        if method == "POST" and url.endswith("/v0/agents"):
            return cursor_ci.HttpResponse(self.launch_status, json.dumps(self.launch_body).encode())
        if method == "GET" and url.endswith("/v0/agents/bc_test"):
            status = self.statuses.pop(0) if self.statuses else "RUNNING"
            return cursor_ci.HttpResponse(
                200, json.dumps({"id": "bc_test", "status": status}).encode()
            )
        if method == "POST" and url.endswith("/stop"):
            return cursor_ci.HttpResponse(200, b"{}")
        if method == "GET" and url.endswith("/status"):
            entries: list[dict[str, str]] = []
            if self.commit_state is not None:
                entries.append(
                    {
                        "context": "cursor-ci",
                        "state": self.commit_state,
                        "description": self.description or "",
                        "updated_at": "2026-01-01T00:00:00Z",
                    }
                )
            payload = {"state": self.commit_state or "pending", "statuses": entries}
            return cursor_ci.HttpResponse(200, json.dumps(payload).encode())
        if method == "GET" and "/comments" in url:
            return cursor_ci.HttpResponse(200, json.dumps(self.comments).encode())
        if method == "POST" and url.endswith("/comments"):
            assert body is not None
            self.posted.append(body)
            self.comments.append({"id": 7, "body": json.loads(body)["body"]})
            return cursor_ci.HttpResponse(201, b"{}")
        if method == "PATCH" and "/comments/" in url:
            assert body is not None
            self.posted.append(body)
            return cursor_ci.HttpResponse(200, b"{}")
        return cursor_ci.HttpResponse(500, b"unexpected")


def _prompt(tmp_path: Path) -> Path:
    path = tmp_path / "prompt.md"
    path.write_text("Make no code changes.\n", encoding="utf-8")
    return path


def _env(prompt: Path, **overrides: str) -> dict[str, str]:
    base = {
        "CURSOR_API_KEY": KEY,
        "GITHUB_TOKEN": TOKEN,
        "CURSOR_CI_SHA": SHA,
        "CURSOR_CI_REPOSITORY_URL": "https://github.com/artificial-hedge/dipcatcher.git",
        "CURSOR_CI_HEAD_REPO": "artificial-hedge/dipcatcher",
        "CURSOR_CI_BASE_REPO": "artificial-hedge/dipcatcher",
        "CURSOR_CI_PR_NUMBER": "42",
        "CURSOR_CI_EVENT": "pull_request",
        "CURSOR_CI_REF_NAME": "feature",
        "CURSOR_CI_TIMEOUT_SECONDS": "30",
        "CURSOR_CI_POLL_SECONDS": "5",
        "CURSOR_CI_PROMPT": str(prompt),
    }
    base.update(overrides)
    return base


def _comment(token: str) -> dict[str, Any]:
    return {"id": 3, "body": f"<!-- cursor-ci -->\n\nCURSOR_CI_RESULT: {token}\n"}


def _run(env: dict[str, str], api: Api, clock: Clock | None = None) -> int:
    timer = clock or Clock()
    return int(
        cursor_ci.run(
            env,
            http=api,
            sleep=timer.sleep,
            now=timer.now,
            handle_signals=False,
        )
    )


def test_launch_body_uses_sha_and_does_not_open_a_pr() -> None:
    body = cursor_ci.build_launch_body(
        repository="https://github.com/artificial-hedge/dipcatcher.git",
        ref=SHA,
        prompt="Make no code changes.\n",
    )
    assert body["source"] == {
        "repository": "https://github.com/artificial-hedge/dipcatcher",
        "ref": SHA,
    }
    assert body["target"] == {"autoCreatePr": False}
    assert "prUrl" not in json.dumps(body)


def test_launch_body_rejects_bad_sha_and_credentials() -> None:
    with pytest.raises(cursor_ci.CursorCIError):
        cursor_ci.build_launch_body(
            repository="https://github.com/artificial-hedge/dipcatcher",
            ref="abc",
            prompt="x",
        )
    with pytest.raises(cursor_ci.CursorCIError):
        cursor_ci.normalize_repository_url("https://user:token@github.com/a/b")


def test_result_token_requires_exactly_one_marker() -> None:
    assert cursor_ci.result_token("CURSOR_CI_RESULT: pass") == "pass"
    assert cursor_ci.result_token("CURSOR_CI_RESULT: fail") == "fail"
    assert cursor_ci.result_token("CURSOR_CI_RESULT: pass\nCURSOR_CI_RESULT: fail") is None
    assert cursor_ci.result_token("looks fine") is None


def test_redact_removes_key_and_basic_encoding() -> None:
    encoded = base64.b64encode(f"{KEY}:".encode()).decode()
    text = f"Authorization: Basic {encoded} key={KEY}"
    redacted = cursor_ci.redact(text, [KEY])
    assert KEY not in redacted
    assert encoded not in redacted


def test_assemble_prompt_appends_context_without_formatting() -> None:
    template = "literal {not_a_field}\n"
    path_settings = cursor_ci.Settings(
        api_key=KEY,
        github_token=TOKEN,
        sha=SHA,
        repository_url="https://github.com/artificial-hedge/dipcatcher",
        head_repo="artificial-hedge/dipcatcher",
        base_repo="artificial-hedge/dipcatcher",
        pr_number="42",
        ref_name="feature",
        event_name="pull_request",
        timeout_s=30,
        poll_s=5,
        prompt_path=Path("prompt.md"),
    )
    text = cursor_ci.assemble_prompt(template, path_settings)
    assert "{not_a_field}" in text
    assert SHA in text
    assert "head sha:" in text


def test_pass_requires_finished_status_and_one_comment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = Api(
        ["RUNNING", "FINISHED"],
        comments=[_comment("pass")],
    )
    code = _run(_env(_prompt(tmp_path)), api)
    captured = capsys.readouterr()
    assert code == 0
    assert "cursor-ci: pass" in captured.out
    assert KEY not in captured.out
    assert KEY not in captured.err
    assert TOKEN not in captured.out
    launch = next(
        body
        for method, url, _headers, body in api.calls
        if method == "POST" and url.endswith("/v0/agents")
    )
    assert launch is not None
    payload = json.loads(launch)
    assert payload["target"]["autoCreatePr"] is False
    assert payload["source"]["ref"] == SHA
    assert "prUrl" not in payload
    assert KEY not in launch.decode()
    auth = next(
        headers["Authorization"] for _m, url, headers, _b in api.calls if url.endswith("/v0/agents")
    )
    assert auth == "Basic " + base64.b64encode(f"{KEY}:".encode()).decode()
    assert not api.posted


def test_reported_ci_failure_fails_without_overwriting_the_comment(tmp_path: Path) -> None:
    api = Api(
        ["FINISHED"],
        commit_state="failure",
        description="CURSOR_CI_RESULT: fail",
        comments=[_comment("fail")],
    )
    assert _run(_env(_prompt(tmp_path)), api) == 1
    assert api.posted == []


def test_missing_status_fails_and_posts_one_comment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = Api(["FINISHED"], commit_state=None, description=None, comments=[])
    code = _run(_env(_prompt(tmp_path)), api)
    captured = capsys.readouterr()
    assert code == 1
    assert "missing commit status" in captured.err
    assert KEY not in captured.err
    assert len(api.posted) == 1
    posted = json.loads(api.posted[0])["body"]
    assert posted.count("<!-- cursor-ci -->") == 1
    assert posted.count("CURSOR_CI_RESULT:") == 1
    assert "CURSOR_CI_RESULT: fail" in posted


def test_timeout_stops_the_agent(tmp_path: Path) -> None:
    api = Api(["RUNNING"])
    clock = Clock()
    code = _run(
        _env(_prompt(tmp_path), CURSOR_CI_TIMEOUT_SECONDS="5", CURSOR_CI_POLL_SECONDS="5"),
        api,
        clock,
    )
    assert code == 1
    assert any(url.endswith("/stop") for _m, url, _h, _b in api.calls)


def test_missing_secret_does_not_call_the_api(tmp_path: Path) -> None:
    api = Api(["FINISHED"])
    env = _env(_prompt(tmp_path))
    del env["CURSOR_API_KEY"]
    assert _run(env, api) == 1
    assert api.calls == []


def test_api_error_and_unparseable_launch_fail(tmp_path: Path) -> None:
    denied = Api(["FINISHED"], launch_status=401, launch_body={})
    # launch_status short-circuits before JSON shape; body still encoded when status is 401
    # because the fake returns launch_body for any launch_status. Force a non-json body.
    assert _run(_env(_prompt(tmp_path)), denied) == 1

    class Bad(Api):
        def __call__(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            body: bytes | None,
            timeout: float,
        ) -> Any:
            if method == "POST" and url.endswith("/v0/agents"):
                self.calls.append((method, url, headers, body))
                return cursor_ci.HttpResponse(200, b"not-json")
            return super().__call__(method, url, headers=headers, body=body, timeout=timeout)

    assert _run(_env(_prompt(tmp_path)), Bad(["FINISHED"])) == 1


def test_agent_error_and_disagreement_fail(tmp_path: Path) -> None:
    error = Api(
        ["ERROR"],
        comments=[_comment("pass")],
    )
    assert _run(_env(_prompt(tmp_path)), error) == 1
    mismatch = Api(
        ["FINISHED"],
        commit_state="success",
        description="CURSOR_CI_RESULT: fail",
        comments=[_comment("pass")],
    )
    assert _run(_env(_prompt(tmp_path)), mismatch) == 1


def test_push_without_a_pr_does_not_require_a_comment(tmp_path: Path) -> None:
    api = Api(["FINISHED"])
    code = _run(_env(_prompt(tmp_path), CURSOR_CI_PR_NUMBER=""), api)
    assert code == 0
    assert not any("/comments" in url for _m, url, _h, _b in api.calls)


def test_two_marker_comments_fail_closed(tmp_path: Path) -> None:
    api = Api(
        ["FINISHED"], comments=[_comment("pass"), {"id": 4, "body": _comment("pass")["body"]}]
    )
    assert _run(_env(_prompt(tmp_path)), api) == 1


def test_main_rejects_arguments() -> None:
    assert cursor_ci.main(["--help"]) == 1


def test_prompt_file_states_the_contract() -> None:
    text = (_ROOT / ".github" / "cursor-ci" / "prompt.md").read_text(encoding="utf-8")
    for phrase in (
        "Make no code changes",
        "Do not commit",
        "ruff check",
        "ruff format --check",
        "uv run mypy src/quant_fund",
        "pytest -n 2",
        "not network",
        "CURSOR_CI_RESULT: pass",
        "CURSOR_CI_RESULT: fail",
        "context='cursor-ci'",
        "gate_pins.sig",
        "corpus_epoch_",
        "src/fx1/serve/api.py",
    ):
        assert phrase in text
    assert "ghp_" in text  # named only as a redact prefix
    assert "github_pat_" in text


def test_cursor_ci_workflow_is_lightweight_and_fail_closed() -> None:
    text = (_ROOT / ".github" / "workflows" / "cursor-ci.yml").read_text(encoding="utf-8")
    assert "runs-on: instabox-large" in text
    assert "cancel-in-progress: true" in text
    assert "group: cursor-ci-${{ github.ref }}" in text
    assert "types: [opened, synchronize, reopened]" in text
    assert "branches: [main]" in text
    assert "workflow_dispatch:" in text
    assert "secrets.CURSOR_API_KEY" in text
    assert "scripts/ci/cursor_ci.py" in text
    assert "timeout-minutes: 100" in text
    assert "CURSOR_CI_TIMEOUT_SECONDS" in text
    assert KEY not in text
    assert "autoCreatePr" not in text  # the script sets this; the workflow must not embed a key


def test_legacy_heavy_jobs_are_gated_and_keep_instabox() -> None:
    for name in GATED_WORKFLOWS:
        text = (_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        jobs = _jobs(text)
        assert jobs, name
        for job, block in jobs.items():
            condition = _job_if(block)
            expected = SPECIAL_JOBS.get((name, job))
            if expected is not None:
                assert expected in condition, (name, job, condition)
                continue
            if name == "docs.yml" and job == "deploy":
                assert "USE_CURSOR_CI" not in condition
                assert "github.ref == 'refs/heads/main'" in condition
                continue
            if name == "simtest.yml" and job == "simtest-large":
                assert condition == "github.event_name == 'workflow_dispatch'"
                continue
            if name == "ci.yml" and job == "refresh-test-durations":
                assert condition == "github.event_name == 'workflow_dispatch' && inputs.suite == 'full'"
                assert "instabox-large" in block
                continue
            assert "USE_CURSOR_CI" in condition, (name, job)
            assert "vars.USE_CURSOR_CI == 'false'" in condition
            if name == "matrix.yml":
                assert "runs-on: ${{ matrix.os }}" in block, job
            else:
                assert "instabox-large" in block, (name, job)


def test_platform_workflows_stay_on_the_runner() -> None:
    for name in UNTOUCHED_WORKFLOWS:
        text = (_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "USE_CURSOR_CI" not in text
        assert "instabox-large" in text


def test_docs_say_where_the_secret_goes() -> None:
    text = (_ROOT / "docs" / "CURSOR_CI.md").read_text(encoding="utf-8")
    assert "CURSOR_API_KEY" in text
    assert "Settings → Secrets and variables → Actions → Secrets" in text
    assert "USE_CURSOR_CI" in text
    assert "never" in text.lower() or "Do not commit" in text
