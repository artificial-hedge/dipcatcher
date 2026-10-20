"""Concierge model client, /models, budgets, cancel, and the headless read API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from fx1.honesty import FORBIDDEN_HEADLINE_TOKENS
from fx1.interactive.concierge import Concierge, run_console
from fx1.interactive.model_client import ModelClient, ModelIdentity, RetryPolicy
from fx1.interactive.model_ux import footer_lines, startup_lines
from fx1.interactive.usage_budget import UsageLimits
from fx1.serve.backends import ToolCompletion
from tests.fx1.stub_model_server import Reply, StubModelServer
from tests.fx1.test_concierge import FakeBackend, make_concierge

KEY_FX1 = "fx1-test-key-aaaa"
KEY_LITE = "lite-test-key-bbbb"
_TINY_RETRY = RetryPolicy(
    max_attempts=2,
    base_delay_s=0.0,
    max_delay_s=0.0,
    jitter=0.0,
    honor_retry_after_up_to_s=30.0,
)


def _factory(**overrides: Any) -> Any:
    def build(**kwargs: Any) -> ModelClient:
        options: dict[str, Any] = {
            "retry": _TINY_RETRY,
            "sleep": lambda _delay: None,
            "timeout_s": 2.0,
        }
        options.update(overrides)
        options.update(kwargs)
        return ModelClient(**options)

    return build


def _joined(lines: list[str]) -> str:
    return "\n".join(lines)


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return tmp_path


def _store_both(srv: StubModelServer) -> None:
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", KEY_FX1, srv.url)
    profiles.set_endpoint("fx1-lite", KEY_LITE, srv.url)


def test_both_models_footer_models_table_and_use_continuity(env: Path) -> None:
    del env
    from fx1.interactive import profiles

    out: list[str] = []
    with StubModelServer({"fx1": KEY_FX1, "fx1-lite": KEY_LITE}) as srv:
        _store_both(srv)
        concierge = Concierge(
            backend_factory=_factory(), output_fn=out.append, animated=False, input_fn=_refuse_input
        )
        concierge.handle_line("hello from fx1")
        concierge.handle_line("/use fx1-lite")
        concierge.handle_line("hello from lite")
        concierge.handle_line("/models")
        requests = srv.requests
    text = _joined(out)
    assert "stub fx1 reply" in text
    assert "stub fx1-lite reply" in text
    assert "· fx1 ·" in text and "· fx1-lite ·" in text
    assert "tokens" in text
    assert "\x1b" not in text
    assert KEY_FX1 not in text and KEY_LITE not in text
    assert requests[0].model == "fx1"
    assert requests[-1].model == "fx1-lite"
    assert requests[0].auth_fingerprint != requests[-1].auth_fingerprint
    second = requests[-1].body
    assert second is not None
    roles = [message["role"] for message in second["messages"]]
    assert roles[0] == "system"
    assert roles.count("system") == 1
    assert "active model changed from fx1 to fx1-lite" in second["messages"][0]["content"]
    assert any(message["role"] == "system" for message in concierge.messages())
    assert "fx1" in text and "fx1-lite" in text
    assert "stored" in text
    assert profiles.fingerprint(KEY_LITE) in text
    assert concierge.active_model == "fx1-lite"
    assert concierge.last_answer == "stub fx1-lite reply"
    assert concierge.last_status == "ok"
    assert concierge.last_usage is not None
    assert concierge.last_identity is not None
    assert concierge.last_identity.served == "fx1-lite"
    assert concierge.last_latency_s is not None


def test_usage_footer_with_and_without_reported_usage(env: Path) -> None:
    del env
    out: list[str] = []
    with StubModelServer({"fx1": KEY_FX1}) as srv:
        from fx1.interactive import profiles

        profiles.set_endpoint("fx1", KEY_FX1, srv.url)
        srv.queue(Reply.ok("pinball loss is 0.10", usage=None, served_model="fx1"))
        srv.queue(
            Reply.ok(
                "pinball loss is 0.20",
                usage={"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7},
                served_model="fx1",
            )
        )
        concierge = Concierge(
            backend_factory=_factory(), output_fn=out.append, animated=False, input_fn=_refuse_input
        )
        concierge.handle_line("first")
        missing = list(out)
        concierge.handle_line("second")
    assert "tokens unreported" in _joined(missing)
    assert concierge.last_usage == {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}
    assert "7 tokens" in _joined(out)
    footer = next(line for line in out if line.startswith("· "))
    for token in FORBIDDEN_HEADLINE_TOKENS:
        assert token not in footer.casefold()


def test_identity_mismatch_warns_and_keeps_the_requested_id(env: Path) -> None:
    del env
    out: list[str] = []
    with StubModelServer({"fx1": KEY_FX1}) as srv:
        from fx1.interactive import profiles

        profiles.set_endpoint("fx1", KEY_FX1, srv.url)
        srv.queue(
            Reply.ok(
                "pinball loss is 0.30",
                usage={"total_tokens": 4},
                served_model="not-fx1",
            )
        )
        concierge = Concierge(
            backend_factory=_factory(), output_fn=out.append, animated=False, input_fn=_refuse_input
        )
        concierge.handle_line("who answered")
    text = _joined(out)
    assert "· fx1 ·" in text
    assert "served not-fx1" in text
    assert "differs from requested model fx1" in text
    assert concierge.active_model == "fx1"
    assert concierge.last_identity is not None
    assert concierge.last_identity.matched is False
    assert concierge.last_identity.served == "not-fx1"


@pytest.mark.parametrize(
    ("reply", "needle", "status"),
    [
        (
            Reply.error(
                401,
                body={
                    "error": {
                        "message": "rejected",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            ),
            "/keys set fx1",
            "error",
        ),
        (
            Reply.error(
                404,
                body={
                    "error": {
                        "message": "no such model",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            ),
            "/models",
            "error",
        ),
        (
            Reply.error(
                429,
                headers={"Retry-After": "1"},
                body={
                    "error": {
                        "message": "slow down",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
            ),
            "rate-limiting",
            "error",
        ),
        (
            Reply.error(
                500,
                body={"error": {"message": "boom", "type": "server_error", "code": "internal"}},
            ),
            "model server failed",
            "error",
        ),
        (Reply.malformed(), "not a valid chat completion", "error"),
    ],
)
def test_model_error_kinds_roll_back(
    env: Path,
    capsys: pytest.CaptureFixture[str],
    reply: Reply,
    needle: str,
    status: str,
) -> None:
    del env
    out: list[str] = []
    with StubModelServer({"fx1": KEY_FX1}) as srv:
        from fx1.interactive import profiles

        profiles.set_endpoint("fx1", KEY_FX1, srv.url)
        srv.queue(reply)
        srv.queue(reply)
        concierge = Concierge(
            backend_factory=_factory(), output_fn=out.append, animated=False, input_fn=_refuse_input
        )
        assert concierge.handle_line("hello") is False
    err = capsys.readouterr().err
    assert needle in err
    assert KEY_FX1 not in err
    assert "<<<not-json>>>" not in err
    assert concierge.messages() == []
    assert concierge.last_status == status


def test_timeout_renders_without_waiting_out_the_server(env: Path) -> None:
    del env
    with StubModelServer({"fx1": KEY_FX1}) as srv:
        from fx1.interactive import profiles

        profiles.set_endpoint("fx1", KEY_FX1, srv.url)
        srv.queue(Reply.delay(2.0, Reply.ok("too late", served_model="fx1")))
        concierge = Concierge(
            backend_factory=_factory(
                timeout_s=0.12,
                retry=RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0),
            ),
            output_fn=lambda *_args, **_kwargs: None,
            animated=False,
            input_fn=_refuse_input,
        )
        concierge.handle_line("hello")
    assert concierge.last_status == "error"
    assert concierge.messages() == []


def test_keyboard_interrupt_cancels_the_turn_and_keeps_the_session(
    env: Path, tmp_path: Path
) -> None:
    del env

    class InterruptBackend:
        supports_cancel = True

        def complete_with_tools(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> ToolCompletion:
            del messages
            event = kwargs.get("cancel_event")
            assert event is not None
            holder[0].cancel_turn()
            assert event.is_set()
            assert "deadline" in kwargs
            raise KeyboardInterrupt

    holder: list[Concierge] = []
    concierge, out = make_concierge(tmp_path)
    holder.append(concierge)
    concierge._backend_factory = lambda **kwargs: InterruptBackend()  # noqa: SLF001
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key", "https://models.example.test/v1")
    concierge._messages.append({"role": "user", "content": "keep me"})  # noqa: SLF001
    assert concierge.handle_line("in flight") is False
    assert concierge.last_status == "cancelled"
    assert concierge.handle_line("/model") is False
    assert "turn cancelled" in _joined(out)
    assert "active model: fx1" in _joined(out)
    assert concierge.messages() == [{"role": "user", "content": "keep me"}]


def test_call_budget_stops_before_a_second_model_call(env: Path, tmp_path: Path) -> None:
    del env
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "flash_retrieve", "arguments": '{"query": "rates"}'},
                    },
                ),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="should not run", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend)
    concierge._usage_preset = UsageLimits(max_model_calls=1)  # noqa: SLF001
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key", "https://models.example.test/v1")
    concierge.handle_line("look up rates")
    assert len(backend.tool_calls_seen) == 1
    assert concierge.last_status == "budget"
    assert "unit price is unknown" in _joined(out)
    assert "no spend cap" in _joined(out)
    assert "should not run" not in _joined(out)


def test_token_budget_stops_after_the_reported_usage(env: Path, tmp_path: Path) -> None:
    del env
    backend = FakeBackend(
        [ToolCompletion(content="pinball loss is 0.4", tool_calls=None, finish_reason="stop")]
    )
    backend.last_usage = {"total_tokens": 9}  # type: ignore[attr-defined]
    concierge, out = make_concierge(tmp_path, backend=backend)
    concierge._usage_preset = UsageLimits(max_total_tokens=1)  # noqa: SLF001
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key", "https://models.example.test/v1")
    concierge.handle_line("score")
    assert concierge.last_status == "budget"
    assert concierge.last_answer == "pinball loss is 0.4"
    assert "unit price is unknown" in _joined(out)


def test_use_without_an_endpoint_warns_and_keeps_the_conversation(
    env: Path, tmp_path: Path
) -> None:
    del env
    concierge, out = make_concierge(tmp_path)
    concierge._messages.append({"role": "user", "content": "prior"})  # noqa: SLF001
    concierge.handle_line("/use fx1-lite")
    assert concierge.active_model == "fx1-lite"
    assert concierge.messages()[0]["content"] == "prior"
    assert "fx1-lite has no endpoint configured — /keys set fx1-lite" in _joined(out)
    assert "conversation kept" in _joined(out)


def test_commands_do_not_mutate_os_environ(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del env
    monkeypatch.setenv("MOONSHOT_API_KEY", "sentinel-key-value-1234")
    monkeypatch.setenv("FX1_BASE_URL", "https://sentinel.example/v1")
    before = dict(os.environ)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "stored-private-key-zzzz")
    backend = FakeBackend(
        [ToolCompletion(content="pinball loss is 0.5", tool_calls=None, finish_reason="stop")]
    )
    concierge, _out = make_concierge(tmp_path, backend=backend)
    concierge._input = lambda prompt: "https://models.example.test/v1"  # noqa: SLF001
    concierge.handle_line("/keys set fx1")
    concierge.handle_line("/use fx1-lite")
    concierge.handle_line("/use fx1")
    concierge.handle_line("hello")
    assert dict(os.environ) == before
    assert "sentinel-key-value-1234" not in json.dumps(concierge.messages())


def test_startup_shows_source_fingerprint_and_invalid_url_safely(env: Path, tmp_path: Path) -> None:
    del env
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1-lite", "test-private-key", "https://lite.example.test/v1")
    out: list[str] = []
    concierge, _cout = make_concierge(tmp_path)
    concierge.model = "fx1-lite"
    run_console(input_fn=_eof, output_fn=out.append, concierge_factory=lambda: concierge)
    text = _joined(out)
    assert "lite.example.test" in text
    assert "endpoint source stored" in text
    assert profiles.fingerprint("test-private-key") in text
    assert "test-private-key" not in text
    assert "\x1b" not in text

    path = profiles.store_path()
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "profiles": {
                    "fx1": {
                        "api_key": "stored-key-value",
                        "base_url": "file:///etc/passwd",
                        "service_origin": "file://",
                        "set_at": "2026-01-01T00:00:00+00:00",
                    }
                },
            }
        )
    )
    path.chmod(0o600)
    lines = "\n".join(startup_lines("fx1"))
    assert "stored-key-value" not in lines
    assert "file://" not in lines
    assert "invalid" in lines.casefold()


def test_help_lists_models_and_approvals(env: Path, tmp_path: Path) -> None:
    del env
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/help")
    text = _joined(out)
    assert "/models" in text
    assert "/approvals" in text


def test_footer_color_respects_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    colored = footer_lines("fx1", 0.01, {"total_tokens": 3}, None)
    assert "\x1b" in colored[0]
    monkeypatch.setenv("NO_COLOR", "1")
    plain = footer_lines(
        "fx1",
        0.2,
        None,
        ModelIdentity("fx1", "other", None, "host", None, False),
    )
    assert all("\x1b" not in line for line in plain)
    assert plain[0].startswith("· fx1 ·")
    assert "served other" in plain[0]
    assert "differs" in plain[1]


def test_headless_reads_are_copies(env: Path, tmp_path: Path) -> None:
    del env
    from fx1.interactive.superpower import SuperpowerResult

    backend = FakeBackend(
        [ToolCompletion(content="pinball loss is 0.6", tool_calls=None, finish_reason="stop")]
    )
    concierge, _out = make_concierge(tmp_path, backend=backend)
    assert concierge.last_status == "none"
    assert concierge.last_answer == ""
    assert concierge.last_result is None
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key", "https://models.example.test/v1")
    concierge.handle_line("score please")
    copied = concierge.messages()
    copied.append({"role": "user", "content": "mutated"})
    assert concierge.messages() != copied
    assert concierge.last_answer == "pinball loss is 0.6"
    concierge.handle_line("/superpower run the benchmark family --sync")
    assert isinstance(concierge.last_result, SuperpowerResult)
    assert concierge.last_answer == concierge.last_result.summary


def test_missing_endpoint_status(env: Path, tmp_path: Path) -> None:
    del env
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("hello")
    assert concierge.last_status == "no_endpoint"
    assert "no model endpoint configured" in _joined(out)


def _refuse_input(prompt: str) -> str:
    raise AssertionError(f"input called: {prompt}")


def _eof(prompt: str) -> str:
    del prompt
    raise EOFError
