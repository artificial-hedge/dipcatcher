"""Concierge console: slash dispatch, background jobs, chat tool loop.

Offline: backend, harness, researcher, store, and approvals are injected.
The chat tool loop is driven by a scripted fake backend so tool execution,
approval gating, and message bookkeeping are all observable.
"""

import io
import json
import os
import threading
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fx1.flash.store import FlashStore
from fx1.interactive import concierge as concierge_module
from fx1.interactive import profiles
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import BackgroundJob, Concierge, run_console
from fx1.interactive.families import goal_lookup_hashes
from fx1.serve.backends import ToolCompletion
from fx1.webresearch.engine import ResearchReport, SourceDoc

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeBackend:
    """Scripted complete_with_tools responses; records every call."""

    def __init__(self, script: list[ToolCompletion]) -> None:
        self.script = list(script)
        self.tool_calls_seen: list[list[dict[str, Any]]] = []
        self.complete_calls: list[list[dict[str, Any]]] = []

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ToolCompletion:
        self.tool_calls_seen.append(list(messages))
        return self.script.pop(0)

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.complete_calls.append(list(messages))
        return "plain model answer"


class FakeHarnessResult:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


_CATALOG_DOCUMENT: dict[str, Any] = {
    "schema_version": 2,
    "catalog_version": 2,
    "receipt_schema_version": 2,
    "required": ["conformal", "volatility", "tail", "ranking"],
    "live_optional": ["candle_order_book"],
    "live_optional_total": 1,
    "exploratory_count": 0,
    "retired_count": 7529,
    "order": ["ranking", "volatility", "conformal", "tail"],
    "forbidden_metric_keys": ["sharpe"],
    "runner": {
        "command": "research",
        "selection": "config",
        "execution_constraint": False,
        "note": "per-family execution constraint is not supported yet",
    },
    "generated_at": "2026-10-12T03:00:00Z",
}


def _catalog_stdout(extra_args: list[str] | None) -> str:
    payload = dict(_CATALOG_DOCUMENT)
    if extra_args is None:
        return json.dumps(payload)
    request = extra_args[1]
    omitted = ("a1_homotopy",)
    requested = set(request.split(","))
    payload["lookup"] = {
        "request_sha256": sha256(request.encode("ascii")).hexdigest(),
        "omitted_matches": [
            name for name in omitted if sha256(name.encode("ascii")).hexdigest() in requested
        ],
    }
    return json.dumps(payload)


class FakeHarness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str] | None]] = []

    def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
        self.calls.append((name, extra_args))
        if name == "research-families":
            return FakeHarnessResult(0, _catalog_stdout(extra_args), "")
        if name == "families":
            goal = ""
            args = extra_args or []
            limit = 4
            if "--select" in args:
                goal = args[args.index("--select") + 1].lower()
            if "--max" in args:
                limit = int(args[args.index("--max") + 1])
            selected = [
                {"id": family_id, "executable": True, "qualification": "required"}
                for family_id in ("volatility", "conformal")
                if family_id in goal
            ][:limit]
            payload = {
                "schema": "dipcatcher.family_surface.v1",
                "selected": selected,
                "exclusions": [],
            }
            return FakeHarnessResult(0, json.dumps(payload), "")
        return FakeHarnessResult(0, f"DATA_LABEL=SYNTHETIC\n{name} ran\n", "")


class FakeResearcher:
    """run() waits for cancel (bounded), records redirect() calls."""

    def __init__(self, *, block: bool = False, report: ResearchReport | None = None) -> None:
        self.redirects: list[str] = []
        self.goals: list[str] = []
        self._block = block
        self._report = report

    def redirect(self, new_goal: str) -> None:
        self.redirects.append(new_goal)

    def run(
        self,
        goal: str,
        *,
        budget: Any = None,
        queries: Any = None,
        cancel_event: threading.Event | None = None,
        on_progress: Any = None,
        **kwargs: Any,
    ) -> ResearchReport:
        del kwargs
        self.goals.append(goal)
        if on_progress is not None:
            from fx1.webresearch.engine import ResearchProgress

            on_progress(ResearchProgress(event="started", detail=goal))
        if self._block and cancel_event is not None:
            cancel_event.wait(3.0)
        if self._report is not None:
            return self._report
        return ResearchReport(
            goal=goal,
            cancelled=bool(cancel_event is not None and cancel_event.is_set()),
            sources=[SourceDoc(url="https://a.example/x", title="A", status=200, text="t")],
        )


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config_dir = tmp_path / "fx1-config"
    if os.name == "nt":
        config_dir.mkdir()
        profiles._restrict_windows_acl(config_dir, directory=True)
    monkeypatch.setenv("FX1_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return tmp_path


def make_concierge(
    tmp_path: Path,
    *,
    backend: FakeBackend | None = None,
    researcher: Any = None,
    harness: Any = None,
    approvals: ApprovalGate | None = None,
) -> tuple[Concierge, list[str]]:
    out: list[str] = []
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    concierge = Concierge(
        store=store,
        researcher=researcher if researcher is not None else FakeResearcher(),
        harness=harness if harness is not None else FakeHarness(),
        approvals=approvals if approvals is not None else ApprovalGate(),
        backend_factory=(lambda **kwargs: backend) if backend is not None else None,
        output_fn=out.append,
        animated=False,
    )
    return concierge, out


def joined(out: list[str]) -> str:
    return "\n".join(str(line) for line in out)


# ---------------------------------------------------------------------------
# basic dispatch
# ---------------------------------------------------------------------------


def test_help_and_exit(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    assert concierge.handle_line("/help") is False
    assert "/superpower" in joined(out)
    assert concierge.handle_line("/exit") is True
    assert concierge.handle_line("quit") is True


def test_unknown_command_reports_error(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    concierge, _ = make_concierge(tmp_path)
    assert concierge.handle_line("/definitely-not-a-command") is False  # survives
    assert "unknown command /definitely-not-a-command" in capsys.readouterr().err


def test_unknown_slash_command_suggests_close_match(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    concierge, _ = make_concierge(tmp_path)
    assert concierge.handle_line("/superpwoer verify receipts") is False
    err = capsys.readouterr().err
    assert "unknown command /superpwoer" in err
    assert "did you mean /superpower" in err


@pytest.mark.parametrize("line", ["/quit", "/QUIT", "/?"])
def test_slash_aliases_and_casefold(env: Path, tmp_path: Path, line: str) -> None:
    concierge, out = make_concierge(tmp_path)
    if line.lower() == "/?":
        assert concierge.handle_line(line) is False
        assert "/superpower" in joined(out)
        return
    assert concierge.handle_line(line) is True


def test_keys_unknown_model_is_a_usage_error(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    concierge, _ = make_concierge(tmp_path)
    assert concierge.handle_line("/keys test gpt-4") is False
    assert "unknown model 'gpt-4'" in capsys.readouterr().err


def test_harness_unknown_command_is_a_usage_error(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    concierge, _ = make_concierge(tmp_path)
    assert concierge.handle_line("/harness run not-a-real-command") is False
    err = capsys.readouterr().err
    assert "not a registered harness command" in err
    assert "/harness list" in err


def test_explicit_harness_research_remains_available(env: Path, tmp_path: Path) -> None:
    harness = FakeHarness()
    concierge, out = make_concierge(tmp_path, harness=harness)

    assert concierge.handle_line("/harness run research --config configs/research.yaml") is False

    assert harness.calls == [("research", ["--config", "configs/research.yaml"])]
    assert "DATA_LABEL=SYNTHETIC" in joined(out)


def test_unknown_command_suggests_a_close_slash_name(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    concierge, _ = make_concierge(tmp_path)
    assert concierge.handle_line("/hel") is False
    err = capsys.readouterr().err
    assert "unknown command /hel" in err
    assert "did you mean /help" in err


def test_model_command(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/model")
    assert "active model: fx1" in joined(out)


def test_chat_offline_is_honest(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("hello there")
    text = joined(out)
    assert "no model endpoint configured" in text
    assert "/keys set" in text


@pytest.mark.parametrize(
    "command, model", [("/keys set", "fx1"), ("/keys set fx1-lite", "fx1-lite")]
)
def test_keys_set_prompts_for_masked_key_and_endpoint(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    model: str,
) -> None:
    key_prompts: list[str] = []
    url_prompts: list[str] = []

    def password(prompt: str) -> str:
        key_prompts.append(prompt)
        return "test-private-api-key"

    def endpoint(prompt: str) -> str:
        url_prompts.append(prompt)
        return "https://models.example.test/v1/chat/completions"

    monkeypatch.setattr("getpass.getpass", password)
    concierge, out = make_concierge(tmp_path)
    concierge._input = endpoint  # noqa: SLF001

    assert concierge.handle_line(command) is False

    assert profiles.resolve_endpoint(model) == (
        "test-private-api-key",
        "https://models.example.test/v1/chat/completions",
    )
    assert key_prompts == [f"› {model} API key: "]
    assert len(url_prompts) == 1 and f"{model} endpoint URL" in url_prompts[0]
    assert "models.example.test" in joined(out)
    assert "test-private-api-key" not in joined(out)


def test_keys_set_blank_url_preserves_existing_endpoint(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = "https://configured.example.test/v1/chat/completions"
    profiles.set_endpoint("fx1", "old-private-key", base_url)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "new-private-key")
    concierge, _ = make_concierge(tmp_path)
    concierge._input = lambda prompt: ""  # noqa: SLF001

    concierge.handle_line("/keys set")

    assert profiles.resolve_endpoint("fx1") == ("new-private-key", base_url)


def test_keys_set_invalid_url_keeps_existing_profile(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = profiles.set_endpoint("fx1", "old-private-key", "https://configured.example.test")
    original = path.read_bytes()
    monkeypatch.setattr("getpass.getpass", lambda prompt: "new-private-key")
    concierge, out = make_concierge(tmp_path)
    concierge._input = lambda prompt: "file:///invalid-endpoint"  # noqa: SLF001

    assert concierge.handle_line("/keys set") is False

    assert path.read_bytes() == original
    assert "invalid base URL" in capsys.readouterr().err
    assert "new-private-key" not in joined(out)


@pytest.mark.parametrize(
    "stage, error", [("key", EOFError), ("url", EOFError), ("url", KeyboardInterrupt)]
)
def test_keys_set_cancel_keeps_existing_profile(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error: type[BaseException],
) -> None:
    path = profiles.set_endpoint("fx1", "old-private-key", "https://configured.example.test")
    original = path.read_bytes()

    def cancelled(prompt: str) -> str:
        raise error

    monkeypatch.setattr(
        "getpass.getpass", cancelled if stage == "key" else lambda prompt: "new-private-key"
    )
    concierge, out = make_concierge(tmp_path)
    concierge._input = cancelled  # noqa: SLF001

    assert concierge.handle_line("/keys set") is False

    assert path.read_bytes() == original
    assert "cancelled; no changes saved" in joined(out)
    assert "new-private-key" not in joined(out)


# ---------------------------------------------------------------------------
# chat tool loop
# ---------------------------------------------------------------------------


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_chat_tool_loop_executes_flash_retrieve(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    store_path = tmp_path / "flash" / "entries.jsonl"
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "flash_retrieve", {"query": "inflation"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Answered from memory.", tool_calls=None, finish_reason="stop"),
        ]
    )
    out: list[str] = []
    store = FlashStore(store_path)
    entry = store.add(text="US inflation was 3.4% in August 2026", task="inflation", tags=["macro"])
    concierge = Concierge(
        store=store,
        researcher=FakeResearcher(),
        harness=FakeHarness(),
        backend_factory=lambda **kwargs: backend,
        output_fn=out.append,
        animated=False,
    )
    concierge.handle_line("what do we know about inflation?")
    assert "Answered from memory." in joined(out)
    # the tool result reached the model as a role=tool message
    second_call_messages = backend.tool_calls_seen[1]
    tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "c1"
    assert "US inflation was 3.4%" in tool_msgs[0]["content"]
    # retrieval marked the entry used
    assert store.get(entry.id).uses == 1


def test_chat_flash_retrieve_never_sends_private_findings_to_backend(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    for i in range(4):
        store.add(text=f"inflation PRIVATE_SENTINEL_{i}", task="inflation", private=True)
    public = store.add(text="inflation public summary", task="inflation")
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "flash_retrieve", {"query": "inflation"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Found public context.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge = Concierge(
        store=store,
        researcher=FakeResearcher(),
        harness=FakeHarness(),
        backend_factory=lambda **kwargs: backend,
        output_fn=lambda *args: None,
        animated=False,
    )
    concierge.handle_line("what do we know about inflation?")
    second_call_messages = backend.tool_calls_seen[1]
    assert "PRIVATE_SENTINEL" not in str(backend.tool_calls_seen)
    tool_msg = [message for message in second_call_messages if message.get("role") == "tool"][0]
    assert "inflation public summary" in tool_msg["content"]
    assert store.get(public.id).uses == 1


def test_chat_harness_tool_denied_without_approval(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    harness = FakeHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": "train"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(
                content="Training requires approval.", tool_calls=None, finish_reason="stop"
            ),
        ]
    )
    concierge, out = make_concierge(
        tmp_path, backend=backend, harness=harness, approvals=ApprovalGate()
    )
    concierge.handle_line("train the ranking family")
    assert harness.calls == []  # consequential command never ran
    assert "not approved" in joined(out)
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert "approval denied" in tool_msg["content"]
    assert "command was not run" in tool_msg["content"]


def test_chat_harness_tool_refuses_unscoped_research(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    harness = FakeHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(
                    _tool_call(
                        "c1",
                        "harness_run",
                        {"name": "research", "args": ["--config", "configs/research.yaml"]},
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ToolCompletion(
                content="No research command ran.", tool_calls=None, finish_reason="stop"
            ),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend, harness=harness)

    concierge.handle_line("run the research benchmark")

    assert harness.calls == []
    assert "harness refused: research" in joined(out)
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert "goal-bound /superpower" in tool_msg["content"]
    assert "explicit operator /harness run research" in tool_msg["content"]
    assert "command was not run" in tool_msg["content"]
    tool_description = next(
        tool["function"]["description"]
        for tool in concierge._chat_tools()  # noqa: SLF001
        if tool["function"]["name"] == "harness_run"
    )
    assert "Research cannot be run through this model tool" in tool_description


def test_chat_harness_tool_runs_registered_command(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    harness = FakeHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": "doctor"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Doctor ran.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend, harness=harness)
    concierge.handle_line("check lab health")
    assert harness.calls == [("doctor", [])]
    assert "DATA_LABEL=SYNTHETIC" in joined(out)


def test_chat_harness_tool_withholds_stdout_after_nonzero_exit(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")

    class FailedHarness(FakeHarness):
        def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
            self.calls.append((name, extra_args))
            return FakeHarnessResult(2, "DATA_LABEL=SYNTHETIC\npartial score", "")

    harness = FailedHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": "doctor"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="The command failed.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend, harness=harness)
    concierge.handle_line("check lab health")
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert "failed (exit 2)" in tool_msg["content"]
    assert "output withheld" in tool_msg["content"]
    assert "partial score" not in tool_msg["content"]
    assert "failed (exit 2)" in joined(out)


@pytest.mark.parametrize(
    ("interrupted_flag", "expected_status"),
    [("cancelled", "cancelled"), ("timed_out", "timed out")],
)
def test_chat_harness_tool_withholds_stdout_after_interruption(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_flag: str,
    expected_status: str,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")

    class InterruptedHarness(FakeHarness):
        def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
            self.calls.append((name, extra_args))
            result = FakeHarnessResult(0, "partial score", "")
            setattr(result, interrupted_flag, True)
            return result

    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": "doctor"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(
                content="The check was interrupted.", tool_calls=None, finish_reason="stop"
            ),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend, harness=InterruptedHarness())
    concierge.handle_line("check lab health")
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert expected_status in tool_msg["content"]
    assert "output withheld" in tool_msg["content"]
    assert "partial score" not in tool_msg["content"]
    assert expected_status in joined(out)


def test_chat_harness_tool_reports_success_with_empty_output(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")

    class EmptyHarness(FakeHarness):
        def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
            self.calls.append((name, extra_args))
            return FakeHarnessResult(0, "", "")

    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": "doctor"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="The check completed.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, _ = make_concierge(tmp_path, backend=backend, harness=EmptyHarness())
    concierge.handle_line("check lab health")
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert "completed (exit 0)" in tool_msg["content"]
    assert "no output" in tool_msg["content"]


@pytest.mark.parametrize("args", [None, 2, ["--flag", 2], {"option": "value"}])
def test_chat_harness_tool_refuses_malformed_args(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: Any
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    harness = FakeHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": "doctor", "args": args}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Invalid arguments.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, _ = make_concierge(tmp_path, backend=backend, harness=harness)
    concierge.handle_line("check lab health")
    assert harness.calls == []
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert "args must be an array of strings" in tool_msg["content"]


@pytest.mark.parametrize("raw_args", ["{", "[]"])
def test_chat_tool_refuses_malformed_json_before_flash_write(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_args: str
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "flash_save", "arguments": raw_args},
    }
    backend = FakeBackend(
        [
            ToolCompletion(content=None, tool_calls=(call,), finish_reason="tool_calls"),
            ToolCompletion(content="Invalid arguments.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, _ = make_concierge(tmp_path, backend=backend)
    concierge.handle_line("save this finding")
    assert concierge._store.all() == []  # noqa: SLF001
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert "arguments must be a JSON object" in tool_msg["content"]


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({"text": "valid note", "task": {"unexpected": "object"}}, "task must be a string"),
        ({"text": "valid note", "uncertainty": "critical"}, "uncertainty must be"),
    ],
)
def test_chat_flash_save_refuses_invalid_optional_fields(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, Any],
    reason: str,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "flash_save", arguments),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Invalid arguments.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend)
    concierge.handle_line("save this finding")
    assert concierge._store.all() == []  # noqa: SLF001
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert reason in tool_msg["content"]
    assert "Invalid arguments." in joined(out)


def test_chat_tool_store_failure_reports_unknown_outcome_without_model_error(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "flash_save", {"text": "a finding"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(
                content="Save failed; I will retry later.", tool_calls=None, finish_reason="stop"
            ),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend)

    def fail_write(**fields: Any) -> None:
        del fields
        raise OSError("private storage path details")

    monkeypatch.setattr(concierge._store, "add", fail_write)  # noqa: SLF001
    concierge.handle_line("save this finding")
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert "tool execution failed" in tool_msg["content"]
    assert "outcome unknown" in tool_msg["content"]
    assert "private storage path details" not in tool_msg["content"]
    assert "Save failed; I will retry later." in joined(out)
    assert "model endpoint error" not in joined(out)


@pytest.mark.parametrize(
    ("error_name", "expected_phrase"),
    [
        ("Fx1HonestyError", "withheld by the fx-1 honesty gate"),
        ("OperationCancelled", "model turn cancelled before completion"),
    ],
)
def test_chat_tool_boundary_preserves_honesty_and_cancellation(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_name: str,
    expected_phrase: str,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "flash_save", {"text": "a finding"}),),
                finish_reason="tool_calls",
            )
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend)

    def abort_tool(call: dict[str, Any]) -> str:
        del call
        raise getattr(concierge_module, error_name)("tool interrupted")

    monkeypatch.setattr(concierge, "_execute_tool", abort_tool)
    concierge.handle_line("save this finding")
    assert expected_phrase in joined(out)
    assert len(backend.tool_calls_seen) == 1


@pytest.mark.parametrize("name", ["rm -rf /", "lab"])
def test_chat_harness_tool_refuses_unregistered(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    harness = FakeHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": name}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Refused.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, _ = make_concierge(tmp_path, backend=backend, harness=harness)
    concierge.handle_line("do something dangerous")
    assert harness.calls == []
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert tool_msg["content"] == "unknown_command"


def test_chat_research_tool_name_is_unknown(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    harness = FakeHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "research", {}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="No command ran.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, _ = make_concierge(tmp_path, backend=backend, harness=harness)

    concierge.handle_line("run the research benchmark")

    assert harness.calls == []
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert "unknown tool 'research'" in tool_msg["content"]


def test_chat_flash_save_tool(env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(
                    _tool_call(
                        "c1",
                        "flash_save",
                        {"text": "Fed held rates in September 2026", "task": "rates"},
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Saved.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge = Concierge(
        store=store,
        researcher=FakeResearcher(),
        harness=FakeHarness(),
        backend_factory=lambda **kwargs: backend,
        output_fn=lambda *a, **k: None,
        animated=False,
    )
    concierge.handle_line("remember that the Fed held rates")
    entries = store.all()
    assert len(entries) == 1
    assert entries[0].text == "Fed held rates in September 2026"
    assert "model-saved" in entries[0].tags


def test_chat_backend_fault_rolls_back_and_survives(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")

    class DeadBackend:
        def complete_with_tools(self, messages: list[dict], **kwargs: Any) -> ToolCompletion:
            raise RuntimeError("hosted_k3 endpoint unreachable")

    concierge, _ = make_concierge(tmp_path)
    concierge._backend_factory = lambda **kwargs: DeadBackend()  # noqa: SLF001
    concierge.handle_line("hello")  # must not raise
    assert concierge._messages == []  # failed turn rolled back  # noqa: SLF001


@pytest.mark.parametrize(
    "answer",
    [
        "The strategy achieved Sharpe: 2.35 over the panel.",
        "The model guarantees nothing, but achieved live trading profits.",
        "On synthetic data the model achieves 0.99 recovery accuracy.",
    ],
)
def test_chat_honesty_refusal_preserves_prior_turn_and_can_continue(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    backend = FakeBackend(
        [
            ToolCompletion(content="Ready to help.", tool_calls=None, finish_reason="stop"),
            ToolCompletion(content=answer, tool_calls=None, finish_reason="stop"),
            ToolCompletion(content="Use proper scores.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend)
    concierge.handle_line("hello")
    prior = list(concierge._messages)  # noqa: SLF001

    assert concierge.handle_line("summarize the result") is False
    assert concierge._messages == prior  # noqa: SLF001
    assert "withheld by the fx-1 honesty gate" in joined(out)
    assert answer not in joined(out)
    captured = capsys.readouterr()
    assert answer not in captured.out + captured.err

    assert concierge.handle_line("which scores should we use?") is False
    sent = backend.tool_calls_seen[-1]
    assert sent[0]["role"] == "system"
    assert sent[1:] == [
        *prior,
        {"role": "user", "content": "which scores should we use?"},
    ]
    assert "Use proper scores." in out
    assert out[-1].startswith("· ")


@pytest.mark.parametrize(
    "answer",
    [
        "The held-out CRPS is 0.12 and pinball loss is 0.08.",
        "On SYNTHETIC data the model achieves 0.99 recovery accuracy.",
    ],
)
def test_chat_honesty_accepts_proper_scores_and_labeled_synthetic_results(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    backend = FakeBackend([ToolCompletion(content=answer, tool_calls=None, finish_reason="stop")])
    concierge, out = make_concierge(tmp_path, backend=backend)

    concierge.handle_line("summarize the result")

    assert out[0] == answer
    assert out[1].startswith("· ")
    assert "\x1b" not in "\n".join(out)
    assert concierge._messages[-1] == {"role": "assistant", "content": answer}  # noqa: SLF001


@pytest.mark.parametrize("unsafe_tool_preamble", [False, True])
def test_chat_honesty_gate_covers_tool_turns(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_tool_preamble: bool,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    forbidden = "The strategy achieved Sharpe: 2.35 over the panel."
    script = [
        ToolCompletion(
            content=forbidden if unsafe_tool_preamble else None,
            tool_calls=(_tool_call("c1", "harness_run", {"name": "doctor"}),),
            finish_reason="tool_calls",
        ),
    ]
    if not unsafe_tool_preamble:
        script.append(ToolCompletion(content=forbidden, tool_calls=None, finish_reason="stop"))
    script.append(ToolCompletion(content="Ready again.", tool_calls=None, finish_reason="stop"))
    backend = FakeBackend(script)
    harness = FakeHarness()
    concierge, out = make_concierge(tmp_path, backend=backend, harness=harness)

    concierge.handle_line("check lab health")

    assert harness.calls == ([] if unsafe_tool_preamble else [("doctor", [])])
    assert forbidden not in joined(out)
    assert "withheld by the fx-1 honesty gate" in joined(out)
    retained = list(concierge._messages)  # noqa: SLF001
    if unsafe_tool_preamble:
        assert retained == []
    else:
        assert [message["role"] for message in retained] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert "DATA_LABEL=SYNTHETIC" in retained[2]["content"]
        assert "withheld by the fx-1 honesty gate" in retained[3]["content"]
    assert forbidden not in str(retained)
    concierge.handle_line("try again")
    sent = backend.tool_calls_seen[-1]
    assert sent[0]["role"] == "system"
    assert "system" not in [message["role"] for message in sent[1:]]
    assert sent[1:] == [*retained, {"role": "user", "content": "try again"}]
    assert "Ready again." in out
    assert out[-1].startswith("· ")


def test_chat_honesty_refusal_preserves_completed_flash_save_for_followup(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    forbidden = "The strategy achieved Sharpe: 2.35 over the panel."
    finding = "Use proper scores to compare the forecasts."
    backend = FakeBackend(
        [
            ToolCompletion(content="Ready to help.", tool_calls=None, finish_reason="stop"),
            ToolCompletion(
                content="I will save the research note.",
                tool_calls=(
                    _tool_call(
                        "save1", "flash_save", {"text": finding, "task": "forecast scoring"}
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content=forbidden, tool_calls=None, finish_reason="stop"),
            ToolCompletion(
                content="The note is already saved.", tool_calls=None, finish_reason="stop"
            ),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend)
    concierge.handle_line("hello")
    prior = list(concierge._messages)  # noqa: SLF001

    concierge.handle_line("remember the scoring guidance")

    entries = concierge._store.all()  # noqa: SLF001
    assert len(entries) == 1 and entries[0].text == finding
    retained = list(concierge._messages)  # noqa: SLF001
    assert retained[: len(prior)] == prior
    assert retained[-3]["tool_calls"][0]["id"] == "save1"
    assert retained[-2] == {
        "role": "tool",
        "tool_call_id": "save1",
        "content": f"saved flash entry {entries[0].id}",
    }
    assert "withheld by the fx-1 honesty gate" in retained[-1]["content"]
    assert forbidden not in str(retained) and forbidden not in joined(out)

    concierge.handle_line("was the note saved?")

    sent = backend.tool_calls_seen[-1]
    assert sent[0]["role"] == "system"
    assert sent[1:] == [
        *retained,
        {"role": "user", "content": "was the note saved?"},
    ]
    assert len(concierge._store.all()) == 1  # noqa: SLF001
    assert "The note is already saved." in out
    assert out[-1].startswith("· ")


# ---------------------------------------------------------------------------
# /research background lifecycle
# ---------------------------------------------------------------------------


def test_research_background_completes_and_announces(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(
        report=ResearchReport(
            goal="fed rates",
            started_at="t0",
            finished_at="t1",
            sources=[SourceDoc(url="https://a.example/x", title="A", status=200, text="t")],
        )
    )
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research fed rates --budget-s 30 --max-sources 3")
    assert "running in the background" in joined(out)
    concierge.join_job(timeout_s=5)
    text = joined(out)
    assert "# Web research: fed rates" in text
    assert "HEURISTIC WEB VERIFICATION" in text
    assert researcher.goals == ["fed rates"]


def test_research_sync_mode(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path, researcher=FakeResearcher())
    concierge.handle_line("/research inflation now --sync")
    text = joined(out)
    assert "# Web research: inflation now" in text


def test_research_queries_option(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher()
    concierge, _ = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research goal --queries one|two --sync")
    assert researcher.goals == ["goal"]


def test_cancel_stops_background_research(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(block=True)
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research long task")
    concierge.handle_line("/status")
    assert "running" in joined(out)
    concierge.handle_line("/cancel")
    assert "cancel requested" in joined(out)
    concierge.join_job(timeout_s=5)
    assert "(cancelled)" in joined(out)


def test_second_background_job_cannot_orphan_first(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(block=True)
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research first goal")
    first = concierge._job  # noqa: SLF001
    assert first is not None

    concierge.handle_line("/research second goal")

    assert concierge._job is first  # noqa: SLF001
    assert "already running" in joined(out)
    concierge.handle_line("/cancel")
    concierge.join_job(timeout_s=5)


def test_shutdown_cancels_and_joins_owned_job(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(block=True)
    concierge, _ = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research long task")
    job = concierge._job  # noqa: SLF001
    assert job is not None and job.running

    assert concierge.shutdown(timeout_s=5)
    assert job.cancel_event.is_set()
    assert job.done_event.is_set()
    assert job.thread is not None and not job.thread.is_alive()


def test_redirect_retargets_running_research(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(block=True)
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research old goal")
    concierge.handle_line("/redirect new goal")
    assert researcher.redirects == ["new goal"]
    assert "redirect requested" in joined(out)
    concierge.handle_line("/cancel")
    concierge.join_job(timeout_s=5)


def test_redirect_refuses_other_background_jobs(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher()
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    release = threading.Event()
    job = BackgroundJob(label="superpower", run_fn=lambda *_: release.wait(5)).start()
    concierge._job = job  # noqa: SLF001
    try:
        concierge.handle_line("/redirect another goal")
        assert researcher.redirects == []
        assert "no running background research job to redirect" in joined(out)
    finally:
        release.set()
        assert job.join(1)


def test_status_without_job(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/status")
    assert "no background job" in joined(out)


def test_background_job_error_announced(env: Path, tmp_path: Path) -> None:
    class ExplodingResearcher(FakeResearcher):
        def run(self, goal: str, **kwargs: Any) -> ResearchReport:
            raise RuntimeError("search backend exploded")

    concierge, out = make_concierge(tmp_path, researcher=ExplodingResearcher())
    concierge.handle_line("/research anything --sync")
    assert "failed" in joined(out)
    assert "exploded" in joined(out)


class StalledResearcher:
    """A transport that does not observe the job's cancellation event."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.stopped = threading.Event()

    def run(self, goal: str, **kwargs: Any) -> ResearchReport:
        from fx1.webresearch.engine import ResearchProgress

        self.started.set()
        self.release.wait(5)
        callback = kwargs.get("on_progress")
        if callback is not None:
            callback(ResearchProgress(event="late", detail="late event"))
        self.stopped.set()
        return ResearchReport(goal=goal)


def test_research_sync_wall_budget_bounds_stalled_transport(env: Path, tmp_path: Path) -> None:
    from fx1.interactive.budget import RunBudget

    researcher = StalledResearcher()
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    concierge._wall_budget = RunBudget(max_wall_s=0.1)  # noqa: SLF001
    try:
        concierge.handle_line("/research slow search --sync")
        assert researcher.started.is_set()
        assert concierge.job is not None and not concierge.job.running
        assert concierge.job.cancel_event.is_set()
        assert "[research] failed: BudgetExceeded: budget exceeded: wall" in joined(out)
    finally:
        researcher.release.set()
        assert researcher.stopped.wait(1)
    concierge.tick()
    assert "late event" not in joined(out)


def test_research_cancel_returns_while_transport_is_stalled(env: Path, tmp_path: Path) -> None:
    researcher = StalledResearcher()
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    try:
        concierge.handle_line("/research slow search")
        assert researcher.started.wait(1)
        concierge.handle_line("/cancel")
        concierge.join_job(timeout_s=1)
        assert concierge.job is not None and not concierge.job.running
        assert concierge.job.result.cancelled is True
        assert "cancelled before a complete report was available" in joined(out)
    finally:
        researcher.release.set()
        assert researcher.stopped.wait(1)
    concierge.tick()
    assert "late event" not in joined(out)


@pytest.mark.parametrize("option", ["budget_s", "max_sources"])
def test_research_rejects_nonpositive_limits(env: Path, tmp_path: Path, option: str) -> None:
    concierge, _ = make_concierge(tmp_path)
    with pytest.raises(ValueError, match="must be positive"):
        concierge.start_research("goal", **{option: 0})
    assert concierge.job is None


# ---------------------------------------------------------------------------
# /superpower through the console
# ---------------------------------------------------------------------------


def test_superpower_help_lists_capabilities(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/superpower help")
    text = joined(out)
    assert "available /superpower capabilities" in text
    assert "web_research" in text


def test_superpower_sync_offline_pipeline(env: Path, tmp_path: Path) -> None:
    harness = FakeHarness()
    researcher = FakeResearcher()
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    store.add(
        text="prior conformal benchmark findings on record",
        task="research benchmark",
        tags=["conformal"],
    )
    concierge, out = make_concierge(tmp_path, researcher=researcher, harness=harness)
    concierge._store = store  # noqa: SLF001
    concierge.handle_line("/superpower run the research benchmark --sync")
    text = joined(out)
    assert "plan for: run the research benchmark" in text
    assert "planner: deterministic" in text
    assert "[ran] research" in text  # harness ran and was explained
    # The family catalog is consulted before the research bench runs.
    assert harness.calls == [
        (
            "research-families",
            ["--lookup-hashes", goal_lookup_hashes("run the research benchmark")],
        ),
        ("research", None),
    ]
    assert "family catalog unavailable" not in text
    # planner coordinated: a pure lab-benchmark goal does not trigger web research
    assert researcher.goals == []
    assert "no model endpoint" in text
    assert "evidence status:" in text
    assert any("superpower" in entry.tags for entry in store.all())


def test_superpower_named_family_does_not_run_full_scorecard(env: Path, tmp_path: Path) -> None:
    harness = FakeHarness()
    concierge, out = make_concierge(tmp_path, harness=harness)
    concierge.handle_line("/superpower run the conformal volatility benchmark --sync")
    text = joined(out)
    assert "family 'conformal' — eligible (required)" in text
    assert "excluded: research: named-family execution unavailable" in text
    assert "result: partial" in text
    assert harness.calls == [
        (
            "research-families",
            ["--lookup-hashes", goal_lookup_hashes("run the conformal volatility benchmark")],
        )
    ]


def test_superpower_background_job(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/superpower summarize tulip history")
    assert "running in the background" in joined(out)
    concierge.join_job(timeout_s=10)
    text = joined(out)
    assert "capabilities used and why" in text
    assert "catalog unavailable" in text
    assert "evidence status:" in text


def test_superpower_denies_consequential(env: Path, tmp_path: Path) -> None:
    harness = FakeHarness()
    concierge, out = make_concierge(
        tmp_path, harness=harness, approvals=ApprovalGate(ask_fn=lambda q: False)
    )
    concierge.handle_line("/superpower train and optimize a model --sync")
    assert all(call[0] != "train" for call in harness.calls)
    assert "catalog unavailable" in joined(out)


# ---------------------------------------------------------------------------
# /flash through the console
# ---------------------------------------------------------------------------


def test_flash_add_list_search_correct_remove(env: Path, tmp_path: Path) -> None:
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    concierge, out = make_concierge(tmp_path)
    concierge._store = store  # noqa: SLF001

    concierge.handle_line(
        '/flash add "US inflation was 3.4% in August 2026" --task inflation '
        "--tags macro,us --source https://x.example/rates --uncertainty low"
    )
    text = joined(out)
    assert "saved flash entry" in text
    entry = store.all()[0]
    assert entry.task == "inflation"
    assert entry.tags == ["macro", "us"]
    assert entry.sources[0].url == "https://x.example/rates"
    assert entry.uncertainty == "low"

    out.clear()
    concierge.handle_line("/flash list")
    assert "US inflation" in joined(out)

    out.clear()
    concierge.handle_line("/flash search inflation rate")
    assert f"{entry.id[:8]}" in joined(out)
    assert "score=" in joined(out)

    out.clear()
    concierge.handle_line(f"/flash correct {entry.id} --uncertainty high")
    assert "corrected" in joined(out)
    assert store.get(entry.id).uncertainty == "high"

    out.clear()
    concierge.handle_line(f"/flash show {entry.id}")
    assert '"uncertainty": "high"' in joined(out)

    out.clear()
    concierge.handle_line(f"/flash remove {entry.id}")
    assert "removed" in joined(out)
    assert store.get(entry.id) is None


def test_flash_bad_uncertainty_reports_error(env: Path, tmp_path: Path) -> None:
    concierge, _ = make_concierge(tmp_path)
    concierge.handle_line("/flash add finding --uncertainty catastrophic")
    entries = concierge._store.all()  # noqa: SLF001
    assert entries == []  # rejected, nothing stored


# ---------------------------------------------------------------------------
# console driver
# ---------------------------------------------------------------------------


def test_run_console_processes_scripted_lines(env: Path, tmp_path: Path) -> None:
    lines = iter(["/help", "/model", "exit"])
    out: list[str] = []

    def input_fn(prompt: str) -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    concierge, cout = make_concierge(tmp_path)
    run_console(
        input_fn=input_fn,
        output_fn=out.append,
        concierge_factory=lambda: concierge,
    )
    banner_text = joined(out)
    assert "dipcatcher concierge" in banner_text
    session_text = joined(cout)
    assert "/superpower" in session_text  # /help rendered
    assert "active model: fx1" in session_text  # /model rendered


def test_run_console_eof_exits_cleanly(env: Path, tmp_path: Path) -> None:
    out: list[str] = []

    def input_fn(prompt: str) -> str:
        raise EOFError

    concierge, _ = make_concierge(tmp_path)
    run_console(input_fn=input_fn, output_fn=out.append, concierge_factory=lambda: concierge)
    assert "dipcatcher concierge" in joined(out)


def test_run_console_prints_one_prompt_per_line_across_idle_polls(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class TerminalInput(io.StringIO):
        def isatty(self) -> bool:
            return True

    stdin = TerminalInput("/model\n/exit\n")
    readiness = iter([False, False, True, False, True])
    monkeypatch.setattr(concierge_module.sys, "stdin", stdin)
    monkeypatch.setattr(concierge_module.sys, "platform", "linux")
    monkeypatch.setattr(
        concierge_module.select,
        "select",
        lambda *args: ([stdin] if next(readiness) else [], [], []),
    )
    concierge, out = make_concierge(tmp_path)

    run_console(concierge_factory=lambda: concierge, output_fn=lambda text: None)

    assert capsys.readouterr().out == "dip › dip › "
    assert "active model: fx1" in joined(out)


def test_run_console_windows_without_msvcrt_uses_blocking_input_without_select(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(concierge_module.sys, "platform", "win32")
    monkeypatch.setattr(concierge_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setitem(concierge_module.sys.modules, "msvcrt", None)

    def unexpected_select(*args: Any) -> Any:
        pytest.fail("Windows console stdin cannot be polled with select")

    monkeypatch.setattr(concierge_module.select, "select", unexpected_select)
    lines = iter(["/model", "/exit"])
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        return next(lines)

    concierge, out = make_concierge(tmp_path)
    startup: list[str] = []

    run_console(concierge_factory=lambda: concierge, input_fn=input_fn, output_fn=startup.append)

    assert prompts == ["dip › ", "dip › "]
    assert "active model: fx1" in joined(out)
    assert "background progress updates after you submit a line" in joined(startup)


def test_run_console_windows_tty_polls_msvcrt(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(concierge_module.sys, "platform", "win32")
    monkeypatch.setattr(concierge_module.sys.stdin, "isatty", lambda: True)
    pending = list("/model\r/exit\r")
    msvcrt = ModuleType("msvcrt")
    msvcrt.kbhit = lambda: bool(pending)  # type: ignore[attr-defined]
    msvcrt.getwch = lambda: pending.pop(0)  # type: ignore[attr-defined]
    monkeypatch.setitem(concierge_module.sys.modules, "msvcrt", msvcrt)

    def unexpected_input(prompt: str) -> str:
        pytest.fail(f"Windows TTY should poll msvcrt, not call input({prompt!r})")

    concierge, out = make_concierge(tmp_path)
    startup: list[str] = []
    run_console(
        concierge_factory=lambda: concierge,
        input_fn=unexpected_input,
        output_fn=startup.append,
    )

    assert "active model: fx1" in joined(out)
    assert "background progress updates while you type" in joined(startup)


@pytest.mark.parametrize("configured", [False, True])
def test_run_console_banner_shows_actual_endpoint_and_respects_no_color(
    env: Path, tmp_path: Path, configured: bool
) -> None:
    if configured:
        profiles.set_endpoint("fx1-lite", "test-private-key", "https://lite.example.test")
    concierge, _ = make_concierge(tmp_path)
    concierge.model = "fx1-lite"
    out: list[str] = []

    def input_fn(prompt: str) -> str:
        raise EOFError

    run_console(input_fn=input_fn, output_fn=out.append, concierge_factory=lambda: concierge)

    text = joined(out)
    assert "model fx1-lite" in text
    assert ("lite.example.test" if configured else "unconfigured") in text
    assert ("/keys set fx1-lite" in text) is not configured
    assert "api.moonshot.ai" not in text
    assert "test-private-key" not in text
    assert "\x1b" not in text


def test_use_keeps_prior_turns(env: Path, tmp_path: Path) -> None:
    profiles.set_endpoint("fx1", "test-private-key", "https://fx1.example.test")
    profiles.set_endpoint("fx1-lite", "test-private-key-lite", "https://lite.example.test")
    concierge, out = make_concierge(tmp_path)
    concierge._messages.append({"role": "user", "content": "prior question"})  # noqa: SLF001
    concierge.handle_line("/use fx1-lite")
    assert concierge.model == "fx1-lite"
    assert concierge._messages[0]["content"] == "prior question"  # noqa: SLF001
    assert any("fx1-lite" in message["content"] for message in concierge._messages)  # noqa: SLF001
    assert "conversation kept" in joined(out)
    assert "test-private-key" not in joined(out)


def test_resume_marks_a_running_session_interrupted(env: Path, tmp_path: Path) -> None:
    from fx1.interactive.sessions import SessionStore

    store = SessionStore(tmp_path / "sessions")
    first, _out = make_concierge(tmp_path)
    first._session_store = store  # noqa: SLF001
    first.handle_line("/help")
    session = first._session  # noqa: SLF001
    assert session is not None
    session["job"]["state"] = "running"
    store.save(session)
    second, out = make_concierge(tmp_path)
    second._session_store = store  # noqa: SLF001
    second.handle_line("/resume")
    assert "not restarted" in joined(out)
    assert second._job is None  # noqa: SLF001
    loaded = store.load(str(session["id"]))
    assert loaded["job"]["state"] == "interrupted"
    assert loaded["job"]["resume"] == "manual"


def test_superpower_skips_research_without_an_eligible_family(env: Path, tmp_path: Path) -> None:
    harness = FakeHarness()
    concierge, out = make_concierge(tmp_path, harness=harness)
    concierge.handle_line("/superpower run the benchmark family --sync")
    text = joined(out)
    assert "catalog unavailable" in text
    assert all(call[0] != "research" for call in harness.calls)


def test_chat_scrubs_an_api_key_out_of_the_transcript(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "super-secret-key")
    backend = FakeBackend(
        [ToolCompletion(content="noted super-secret-key", tool_calls=None, finish_reason="stop")]
    )
    concierge, out = make_concierge(tmp_path, backend=backend)
    concierge.handle_line("please keep super-secret-key private")
    blob = json.dumps(concierge._messages)  # noqa: SLF001
    assert "super-secret-key" not in blob
    assert "[redacted]" in blob
    assert "super-secret-key" not in joined(out)


def test_background_job_drain_and_cancel() -> None:
    release = threading.Event()

    def run_fn(cancel: threading.Event, on_event: Any) -> str:
        on_event("working", "step 1")
        cancel.wait(3.0)
        release.set()
        return "done-after-cancel"

    job = BackgroundJob(label="test", run_fn=run_fn).start()
    job.cancel()
    assert job.done_event.wait(5.0)
    assert release.is_set()
    events = job.drain()
    assert ("working", "step 1") in events
    assert job.result == "done-after-cancel"
    assert job.drain() == []  # drained once


def test_budget_command_limits_family_selection(env: Path, tmp_path: Path) -> None:
    del env
    harness = FakeHarness()
    concierge, out = make_concierge(tmp_path, harness=harness)
    concierge.handle_line("/budget max-families 1")
    concierge.handle_line("/superpower run the conformal volatility benchmark --sync")
    text = joined(out)
    assert "budget max_families=1" in text
    assert "catalog unavailable" in text
    assert all(call[0] != "research" for call in harness.calls)
