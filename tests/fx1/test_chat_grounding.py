"""Chat answers may claim verification only for receipts this turn verified."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.interactive.concierge import Concierge
from fx1.interactive.headless import ExecRequest, run_exec
from tests.fx1.redteam_support import completion, isolate, joined, open_chat
from tests.fx1.test_concierge import FakeBackend

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
