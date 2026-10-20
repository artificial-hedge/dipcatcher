"""Scoped, fail-closed approvals and the legacy yes/no gate."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from fx1.interactive.approvals import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovalRequest,
    console_ask_fn,
    console_decide_fn,
)
from fx1.interactive.redact import REDACTED

_SECRET = "sk-" + "a" * 20


def test_legacy_gate_denies_without_a_prompter() -> None:
    gate = ApprovalGate()
    assert gate.ask("do the thing?") is False
    assert gate.denied() == 1
    assert gate.granted() == 0
    assert gate.records[0].question == "do the thing?"
    assert gate.records[0].scope == "no-prompter"
    assert gate.records[0].key is None
    assert gate.records[0].timestamp


def test_legacy_ask_fn_records_once_or_denied() -> None:
    gate = ApprovalGate(ask_fn=lambda question: True)
    assert gate.ask("a?", kind="consequential") is True
    assert gate.granted() == 1
    assert gate.records[0].kind == "consequential"
    assert gate.records[0].scope == "once"
    denied = ApprovalGate(ask_fn=lambda question: False)
    assert denied.ask("no") is False
    assert denied.records[0].scope == "denied"


def test_legacy_record_constructor_defaults() -> None:
    record = ApprovalRecord("action", "q", True, "2026-01-02T03:04:05+00:00")
    assert record.key is None
    assert record.scope == "once"


def test_console_ask_fn_yes_no_and_eof() -> None:
    assert console_ask_fn(lambda prompt: "y")("q?") is True
    assert console_ask_fn(lambda prompt: "yes")("q?") is True
    assert console_ask_fn(lambda prompt: "n")("q?") is False
    assert console_ask_fn(lambda prompt: "")("q?") is False

    def eof(prompt: str) -> str:
        raise EOFError

    assert console_ask_fn(eof)("q?") is False


def test_prompter_exceptions_deny_without_recording_the_error() -> None:
    def boom(question: str) -> bool:
        raise RuntimeError(_SECRET)

    gate = ApprovalGate(ask_fn=boom)
    assert gate.ask("q", key="harness:research") is False
    assert gate.records[0].scope == "denied"
    assert _SECRET not in repr(gate)
    assert _SECRET not in repr(gate.records[0])

    def interrupt(question: str) -> bool:
        raise KeyboardInterrupt

    assert ApprovalGate(ask_fn=interrupt).ask("q") is False

    def stop(question: str) -> bool:
        raise SystemExit(2)

    with pytest.raises(SystemExit):
        ApprovalGate(ask_fn=stop).ask("q")


def test_session_grant_reuses_only_the_same_kind_and_key() -> None:
    prompts = {"n": 0}

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        prompts["n"] += 1
        return ApprovalDecision.SESSION

    gate = ApprovalGate(decide_fn=decide)
    assert gate.ask("q", kind="consequential", key="harness:research") is True
    assert gate.ask("q2", kind="consequential", key="harness:research") is True
    assert prompts["n"] == 1
    assert gate.records[0].scope == "session"
    assert gate.records[1].scope == "session-reuse"
    assert gate.ask("t", kind="consequential", key="harness:train") is True
    assert gate.ask("t", kind="action", key="harness:research") is True
    assert prompts["n"] == 3
    assert gate.granted() == 4


def test_once_and_session_without_a_key_do_not_stick() -> None:
    prompts = {"n": 0}

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        prompts["n"] += 1
        return ApprovalDecision.ONCE

    gate = ApprovalGate(decide_fn=decide)
    assert gate.ask("q", kind="consequential", key="harness:research") is True
    assert gate.ask("q", kind="consequential", key="harness:research") is True
    assert prompts["n"] == 2
    assert gate.records[0].scope == "once"

    prompts["n"] = 0

    def always(request: ApprovalRequest) -> ApprovalDecision:
        prompts["n"] += 1
        return ApprovalDecision.SESSION

    unkeyed = ApprovalGate(decide_fn=always)
    assert unkeyed.ask("q") is True
    assert unkeyed.ask("q") is True
    assert prompts["n"] == 2
    assert unkeyed.records[0].scope == "once"


def test_revoke_drops_one_key_or_every_grant() -> None:
    prompts = {"n": 0}

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        prompts["n"] += 1
        return ApprovalDecision.SESSION

    gate = ApprovalGate(decide_fn=decide)
    gate.ask("q", kind="consequential", key="harness:research")
    gate.ask("t", kind="consequential", key="harness:train")
    gate.revoke("harness:research")
    assert gate.ask("q", kind="consequential", key="harness:research") is True
    assert prompts["n"] == 3
    assert gate.ask("t", kind="consequential", key="harness:train") is True
    assert prompts["n"] == 3
    gate.revoke()
    assert gate.ask("t", kind="consequential", key="harness:train") is True
    assert prompts["n"] == 4


def test_decide_fn_errors_deny() -> None:
    def eof(request: ApprovalRequest) -> ApprovalDecision:
        raise EOFError

    def interrupt(request: ApprovalRequest) -> ApprovalDecision:
        raise KeyboardInterrupt

    assert ApprovalGate(decide_fn=eof).ask("q", key="k") is False
    gate = ApprovalGate(decide_fn=interrupt)
    assert gate.ask("q", key="k") is False
    assert gate.records[0].scope == "denied"


def test_policy_allows_keys_but_not_consequential_by_kind() -> None:
    rules = ApprovalPolicy(allow_keys={"harness:research"}, allow_kinds={"consequential", "read"})
    assert rules.decide(ApprovalRequest("harness:train", "consequential", "q")) is (
        ApprovalDecision.DENY
    )
    assert rules.decide(ApprovalRequest("harness:research", "consequential", "q")) is (
        ApprovalDecision.ONCE
    )
    assert rules.decide(ApprovalRequest("x", "read", "q")) is ApprovalDecision.ONCE
    assert rules.decide(ApprovalRequest("x", "other", "q")) is ApprovalDecision.DENY
    gate = ApprovalGate.non_interactive(rules)
    assert gate.ask("run", kind="consequential", key="harness:research") is True
    assert gate.records[0].scope == "policy"
    assert gate.ask("run", kind="consequential", key="harness:research") is True
    assert gate.records[-1].scope == "policy"
    assert gate.ask("run", kind="consequential", key="harness:train") is False
    assert gate.records[-1].scope == "denied"
    assert gate.ask("read it", kind="read", key="memo") is True


def test_non_interactive_never_calls_a_prompter() -> None:
    calls = {"n": 0}

    def asker(question: str) -> bool:
        calls["n"] += 1
        return True

    rules = ApprovalPolicy(allow_keys={"harness:research"})
    gate = ApprovalGate(ask_fn=asker, policy=rules, interactive=False)
    assert gate.ask("run", kind="consequential", key="harness:research") is True
    assert gate.ask("run", kind="consequential", key="harness:train") is False
    assert calls["n"] == 0
    empty = ApprovalGate.non_interactive(ApprovalPolicy())
    assert empty.ask("q", kind="read", key="x") is False
    assert empty.records[0].scope == "denied"


def test_interactive_prompter_wins_over_policy() -> None:
    rules = ApprovalPolicy(allow_keys={"harness:research"})
    gate = ApprovalGate(ask_fn=lambda question: False, policy=rules)
    assert gate.ask("q", kind="consequential", key="harness:research") is False
    richer = ApprovalGate(
        ask_fn=lambda question: False,
        decide_fn=lambda request: ApprovalDecision.ONCE,
    )
    assert richer.ask("q", key="k") is True
    assert richer.records[0].scope == "once"


def test_snapshot_redacts_and_tracks_reusable_grants() -> None:
    question = "open https://alice:hunter2pw@host.example/v1 please"
    gate = ApprovalGate(decide_fn=lambda request: ApprovalDecision.SESSION)
    gate.ask(question, kind="private_data", key="web:lookup")
    snap = gate.snapshot()
    assert snap[0]["scope"] == "session"
    assert snap[0]["reusable"] is True
    assert snap[0]["approved"] is True
    assert "hunter2pw" not in str(snap)
    assert REDACTED in snap[0]["question"]
    assert "hunter2pw" not in repr(gate)
    assert "hunter2pw" not in repr(gate.records[0])
    snap[0]["question"] = "mutated"
    assert gate.records[0].question == question
    gate.revoke("web:lookup")
    assert gate.snapshot()[0]["reusable"] is False
    assert gate.snapshot()[0]["scope"] == "session"


def test_snapshot_redacts_provider_key_shapes() -> None:
    gate = ApprovalGate(ask_fn=lambda question: True)
    gate.ask(f"use {_SECRET}", key="k")
    assert _SECRET not in str(gate.snapshot())
    assert _SECRET not in repr(gate.records[0])


def test_clock_is_injectable() -> None:
    fixed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    gate = ApprovalGate(ask_fn=lambda question: True, clock=lambda: fixed)
    gate.ask("q")
    assert gate.records[0].timestamp == "2026-01-02T03:04:05+00:00"
    naive = datetime(2026, 1, 2, 3, 4, 5)
    other = ApprovalGate(ask_fn=lambda question: False, clock=lambda: naive)
    other.ask("q")
    assert other.records[0].timestamp.endswith("+00:00")


def test_session_grant_is_thread_safe() -> None:
    prompts = {"n": 0}
    lock = threading.Lock()

    def decide(request: ApprovalRequest) -> ApprovalDecision:
        with lock:
            prompts["n"] += 1
        return ApprovalDecision.SESSION

    gate = ApprovalGate(decide_fn=decide)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=2)
            assert gate.ask("run?", kind="consequential", key="harness:research") is True
        except BaseException as exc:  # noqa: BLE001 - collected for the join assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert errors == []
    assert prompts["n"] == 1
    assert gate.granted() == 8


def test_concurrent_denies_record_every_ask() -> None:
    gate = ApprovalGate(ask_fn=lambda question: False)
    barrier = threading.Barrier(12)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=2)
            assert gate.ask("q", kind="action") is False
        except BaseException as exc:  # noqa: BLE001 - collected for the join assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert errors == []
    assert gate.denied() == 12
    assert len(gate.records) == 12


@pytest.mark.parametrize(
    ("answer", "decision"),
    [
        ("y", ApprovalDecision.ONCE),
        ("YES", ApprovalDecision.ONCE),
        ("a", ApprovalDecision.SESSION),
        ("always", ApprovalDecision.SESSION),
        ("n", ApprovalDecision.DENY),
        ("", ApprovalDecision.DENY),
        ("maybe", ApprovalDecision.DENY),
    ],
)
def test_console_decide_fn_parses_answers(answer: str, decision: ApprovalDecision) -> None:
    request = ApprovalRequest("k", "consequential", "run?")
    seen: list[str] = []

    def read(prompt: str) -> str:
        seen.append(prompt)
        return answer

    assert console_decide_fn(read)(request) is decision
    assert "run?" in seen[0]
    assert "[y]es once" in seen[0]


def test_console_decide_fn_eof_and_custom_output() -> None:
    request = ApprovalRequest("k", "action", "run?")

    def eof(prompt: str) -> str:
        raise EOFError

    def interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    assert console_decide_fn(eof)(request) is ApprovalDecision.DENY
    assert console_decide_fn(interrupt)(request) is ApprovalDecision.DENY
    lines: list[str] = []
    seen: list[str] = []

    def read(prompt: str) -> str:
        seen.append(prompt)
        return "yes"

    assert console_decide_fn(read, lines.append)(request) is ApprovalDecision.ONCE
    assert seen == [""]
    assert "run?" in lines[0]
