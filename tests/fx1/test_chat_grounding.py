"""Chat answers may claim verification only for receipts this turn verified."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import Concierge
from fx1.interactive.headless import ExecRequest, run_exec
from fx1.interactive.sessions import SessionStore
from tests.fx1.redteam_support import completion, isolate, joined, open_chat, tool
from tests.fx1.stub_model_server import StubModelServer
from tests.fx1.test_concierge import FakeBackend, FakeHarnessResult

_KEY = "sk-test-fx1-secret-value-aaa111"


@pytest.fixture()
def root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    isolate(monkeypatch, tmp_path)
    return tmp_path


def test_hostile_verified_prose_is_rewritten(root: Path) -> None:
    concierge, out, _harness = open_chat(root, [completion("This result is verified.")])
    concierge.handle_line("report")
    text = joined(out)
    assert concierge.last_status == "ok"
    assert "unverified" in text.casefold()
    assert "verified." not in text.casefold()


def test_benign_answer_is_unchanged(root: Path) -> None:
    concierge, out, _harness = open_chat(root, [completion("QLIKE is 0.42")])
    concierge.handle_line("report")
    assert concierge.last_status == "ok"
    assert "QLIKE is 0.42" in joined(out)


def test_evidence_error_withholds_without_the_raw_text(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.interactive.evidence import EvidenceError

    def explode(_summary: str, **_kwargs: object) -> str:
        raise EvidenceError("withhold")

    monkeypatch.setattr("fx1.interactive.chat_turn.ground_summary", explode)
    concierge, out, _harness = open_chat(root, [completion("SECRET-PROSE-12345")])
    concierge.handle_line("report")
    text = joined(out)
    assert concierge.last_status == "withheld"
    assert "SECRET-PROSE-12345" not in text
    assert "withheld" in text.casefold()


def test_a_verified_receipt_keeps_the_claim(root: Path) -> None:
    receipt = _sealed_receipt(root)
    claim = f"verified receipt {receipt}"

    class _Harness:
        def run(
            self, name: str, extra_args: list[str] | None = None, **_kwargs: object
        ) -> FakeHarnessResult:
            del name, extra_args
            return FakeHarnessResult(0, f"receipt={receipt}\n")

    script = [
        completion("", (tool("c1", "harness_run", {"name": "doctor", "args": []}),)),
        completion(claim),
    ]
    concierge, out, _harness = open_chat(root, script, harness=_Harness())
    concierge.handle_line("hello")
    text = joined(out)
    assert concierge.last_status == "ok", text
    assert claim in text
    assert "unverified" not in text.casefold()


def test_exec_rewrites_the_same_claim(root: Path) -> None:
    from fx1.interactive import profiles

    profiles.set_endpoint("fx1", "test-key-value", "https://models.example.test/v1")
    backend = FakeBackend([completion("This result is verified.")])
    outcome = run_exec(
        ExecRequest(prompt="report", timeout_s=30),
        backend_factory=lambda **_kwargs: backend,
        install_signals=False,
    )
    assert outcome.status == "ok"
    assert "unverified" in outcome.answer.casefold()
    assert "verified." not in outcome.answer.casefold()


def test_input_cap_blocks_the_model_and_keeps_the_session(root: Path) -> None:
    with StubModelServer({"fx1": _KEY}) as server:
        from fx1.interactive import profiles

        profiles.set_endpoint("fx1", _KEY, server.url)
        store = SessionStore(root / "sessions")
        out: list[str] = []
        concierge = Concierge(
            session_store=store,
            approvals=ApprovalGate(),
            output_fn=out.append,
            animated=False,
            input_fn=lambda _prompt: "",
        )
        assert concierge.handle_line("/help") is False
        session_id = str(concierge._session["id"])  # noqa: SLF001
        counted = len(server.requests)
        assert concierge.handle_line("a" * 16_000) is False
        assert len(server.requests) > counted
        saved = store.load(session_id)
        mid = len(server.requests)
        out.clear()
        assert concierge.handle_line("b" * 16_001) is False
        assert len(server.requests) == mid
        assert concierge.last_status == "error"
        assert "input too long (16001 characters; limit 16000)" in joined(out)
        huge = "/help " + ("y" * 16_000)
        out.clear()
        concierge.handle_line(huge)
        assert "plain text" not in joined(out)
        assert "input too long" in joined(out)
        after = store.load(session_id)
        assert after.get("messages") == saved.get("messages")
        assert "b" * 100 not in json.dumps(after)
        outcome = run_exec(
            ExecRequest(prompt="c" * 16_001, timeout_s=30),
            install_signals=False,
        )
        assert outcome.status == "error"
        blob = json.dumps(outcome.as_json())
        assert "16001" in blob
        assert len(server.requests) == mid


def test_say_error_redacts_a_key(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    del root
    concierge = Concierge(output_fn=lambda _text="": None, animated=False, session_store=None)
    concierge._say_error(RuntimeError(f"failed {_KEY}"))
    err = capsys.readouterr().err
    assert err.startswith("error:") or "error:" in err
    assert _KEY not in err


def test_unknown_command_suggests_and_sanitizes(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del root
    concierge = Concierge(output_fn=lambda _text="": None, animated=False, session_store=None)
    concierge.handle_line("/superpwer")
    err = capsys.readouterr().err
    assert "unknown command /superpwer; did you mean /superpower? try /help" in err
    concierge.handle_line("/zzzzqqqq")
    err = capsys.readouterr().err
    assert "unknown command /zzzzqqqq; try /help" in err
    assert "did you mean" not in err.split("zzzzqqqq", 1)[-1]
    concierge.handle_line("/super\x1b[31mpwer")
    err = capsys.readouterr().err
    assert "\x1b" not in err
    assert "did you mean /superpower?" in err


def test_endpoint_prompt_uses_ascii_when_unicode_is_unavailable(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del root
    seen: list[str] = []

    def _getpass(prompt: str = "") -> str:
        seen.append(prompt)
        return "test-key-value"

    def _input(prompt: str = "") -> str:
        seen.append(prompt)
        return ""

    monkeypatch.setattr("getpass.getpass", _getpass)
    monkeypatch.setattr("fx1.interactive.terminal.unicode_stdout", lambda stream=None: False)
    concierge = Concierge(
        output_fn=lambda _text="": None,
        animated=False,
        session_store=None,
        input_fn=_input,
    )
    concierge._prompt_endpoint("fx1")
    assert seen
    assert all(item.startswith("> ") for item in seen)
    assert all("›" not in item for item in seen)


def _sealed_receipt(root: Path) -> Path:
    from fx1.harness import Harness

    data = root / "vol.csv"
    lines = ["timestamp,realized,forecast_var"]
    for index in range(40):
        lines.append(f"2020-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00+00:00,1.0,1.0")
    data.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = root / "out"
    result = Harness().run(
        "score-forecasts",
        ["--task", "volatility", "--data", str(data), "--out-dir", str(out)],
    )
    assert result.exit_code == 0, result.stderr[-1500:]
    found = list(out.rglob("forecast_scoring_*.json"))
    assert found
    return found[0].resolve()
