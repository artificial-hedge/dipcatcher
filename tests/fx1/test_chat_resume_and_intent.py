"""Plain-text orchestration routing and explicit session resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx1.interactive.concierge import Concierge
from fx1.interactive.intent import looks_like_orchestration
from fx1.interactive.sessions import SessionStore
from fx1.interactive.superpower import (
    CapabilityRun,
    PlannedCapability,
    SuperpowerPlan,
    SuperpowerResult,
)


class _Result:
    def __init__(self, exit_code: int, stdout: str) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = ""


class _Harness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str] | None]] = []

    def run(self, name: str, extra_args: list[str] | None = None) -> _Result:
        self.calls.append((name, extra_args))
        if name == "families":
            payload = {
                "schema": "dipcatcher.family_surface.v1",
                "selected": [
                    {
                        "id": "volatility",
                        "executable": True,
                        "qualification": "required",
                    }
                ],
                "exclusions": [],
            }
            return _Result(0, json.dumps(payload))
        return _Result(0, "DATA_LABEL=SYNTHETIC\nSYNTHETIC\nPARTIAL_FAMILY_RUN verified=false\n")


def test_contract_example_is_orchestration() -> None:
    assert looks_like_orchestration("Check whether my volatility forecasts are calibrated.")
    assert not looks_like_orchestration("/superpower check volatility")
    assert not looks_like_orchestration("hello there")


@pytest.mark.parametrize(
    "text",
    [
        "check volatility forecasts",
        "Please run the volatility benchmark",
        "Can you check whether my volatility forecasts are calibrated?",
        "I want to understand why my CRPS dropped, so run the calibration check on my forecasts",
        "Explain CRPS. Then run the volatility benchmark.",
        "score my conformal forecasts",
    ],
)
def test_requests_to_do_research_route_to_orchestration(text: str) -> None:
    assert looks_like_orchestration(text)


@pytest.mark.parametrize(
    "text",
    [
        "hello there",
        "How do I run a volatility benchmark?",
        "What does it mean to evaluate volatility forecasts?",
        "Why was my CRPS score low on the forecasts?",
        "Explain how to calibrate conformal forecasts",
        "Show me how to run a volatility benchmark",
        "Can I run a benchmark on volatility forecasts?",
    ],
)
def test_questions_about_the_work_stay_on_the_chat_path(text: str) -> None:
    assert not looks_like_orchestration(text)


@pytest.mark.parametrize(
    "text",
    [
        "QLIKE on the variance forecasts please.",
        "Rank IC of the cross section.",
        "PIT histogram for the holdout.",
        "Serial dependence of the probability integral transform.",
        "Doctor preflight on the research tree.",
        "Selective volatility family only.",
        "Alarm rate of the sequential lanes.",
        "Verify the sealed receipt digest.",
        "Full scorecard of required families.",
        "Pinball and CRPS on the quantile file.",
        "GARCH realized variance scoring.",
        "Health of the lab receipt store.",
    ],
)
def test_diagnostic_noun_phrases_route_to_orchestration(text: str) -> None:
    assert looks_like_orchestration(text)


@pytest.mark.parametrize(
    "text",
    [
        "I like pinball",
        "the family is visiting tomorrow",
        "the receipt is a hash, not a request",
        "volatility is a noun in this sentence",
        "what is a scorecard?",
        "how do I read a PIT histogram?",
    ],
)
def test_ordinary_mentions_of_research_words_stay_chat(text: str) -> None:
    assert not looks_like_orchestration(text)


def test_plain_text_uses_the_superpower_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    harness = _Harness()
    out: list[str] = []
    concierge = Concierge(
        harness=harness,
        output_fn=out.append,
        animated=False,
        session_store=SessionStore(tmp_path / "sessions"),
    )
    concierge.handle_line("Check whether my volatility forecasts are calibrated.")
    text = "\n".join(out)
    assert "routing this to research orchestration" in text
    assert harness.calls[0][0] == "capabilities"
    concierge.join_job(timeout_s=10)
    assert "catalog unavailable" in text
    assert all(name != "research" for name, _args in harness.calls)


def test_restore_latest_reloads_messages_without_restarting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    store = SessionStore(tmp_path / "sessions")
    first = Concierge(output_fn=lambda *_a, **_k: None, animated=False, session_store=store)
    first._messages.append({"role": "user", "content": "keep me"})  # noqa: SLF001
    first._persist_session()  # noqa: SLF001
    second = Concierge(
        output_fn=lambda *_a, **_k: None,
        animated=False,
        session_store=SessionStore(tmp_path / "sessions"),
    )
    second.restore_latest()
    assert second._messages[0]["content"] == "keep me"  # noqa: SLF001
    assert second.resume_notice.startswith("resumed session")


def test_evidence_lists_job_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    out: list[str] = []
    concierge = Concierge(output_fn=out.append, animated=False)
    plan = SuperpowerPlan(goal="qlike", capabilities=[PlannedCapability("research", "test")])
    result = SuperpowerResult(goal="qlike", plan=plan)
    result.runs.append(
        CapabilityRun(
            "volatility", "volatility", "ran", "partial", artifacts=["receipts/family.json"]
        )
    )
    from fx1.interactive.concierge import BackgroundJob

    job = BackgroundJob(label="superpower", run_fn=lambda *_a: result)
    job.result = result
    job.done_event.set()
    concierge._job = job  # noqa: SLF001
    concierge.handle_line("/evidence")
    assert "receipts/family.json" in "\n".join(out)


def test_private_flash_stays_out_of_the_model_prompt(tmp_path: Path) -> None:
    from fx1.flash.retrieve import context_block, release_private, retrieve
    from fx1.flash.store import FlashStore

    store = FlashStore(tmp_path / "flash.jsonl")
    store.add(text="desk volatility limit is confidential", task="volatility", private=True)
    hits = retrieve(store, "volatility", k=4)
    assert hits
    released = release_private(hits, None)
    assert released == []
    assert "confidential" not in context_block(released)


def test_unknown_model_slash_does_not_echo_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    concierge = Concierge(output_fn=lambda *_args, **_kwargs: None, animated=False)
    assert concierge.handle_line("/use sk-livesecrettoken") is False
    assert concierge.handle_line("/use not-a-model") is False
    err = capsys.readouterr().err
    assert "sk-livesecrettoken" not in err
    assert "command failed" in err
    assert "not-a-model" in err


def test_chat_resume_flag_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from dipcatcher_cli.app import main
    from fx1.interactive import concierge as concierge_module

    seen: dict[str, object] = {}

    def fake(*, model: str, resume: bool = False, model_explicit: bool = False) -> None:
        seen["model"] = model
        seen["resume"] = resume
        seen["explicit"] = model_explicit

    monkeypatch.setattr(concierge_module, "run_console", fake)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    main(["chat", "--model", "fx1-lite", "--resume"])
    assert seen == {"model": "fx1-lite", "resume": True, "explicit": True}
