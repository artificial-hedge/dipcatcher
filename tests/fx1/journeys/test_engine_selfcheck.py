"""The journey engine fails closed on its own mistakes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.fx1.journeys.engine import (
    SCOPE,
    InvariantView,
    JourneyResult,
    SuiteReport,
    child_pids,
    evaluate,
    gate_problems,
    invariant_problems,
    journey_digest,
    make_harness,
    run_journey,
    unexpected_children,
    write_freeze,
)
from tests.fx1.journeys.schema import (
    CATEGORIES,
    JourneySchemaError,
    Setup,
    load_journeys,
    parse_journey,
)

_HELP = {
    "id": "onboarding-009",
    "category": "onboarding",
    "title": "Inline help still works with no endpoint",
    "critical": False,
    "in_chat": True,
    "setup": {"endpoints": {"fx1": "missing"}},
    "steps": [{"say": "/help", "expect": ["/superpower"], "status": "ok"}],
}


def test_schema_rejects_bad_id_and_out_of_chat_ops() -> None:
    with pytest.raises(JourneySchemaError, match="id"):
        parse_journey({**_HELP, "id": "nope"})
    with pytest.raises(JourneySchemaError, match="shell"):
        parse_journey({**_HELP, "steps": [{"shell": ["echo", "hi"]}]})


def test_duplicate_ids_fail_the_load(tmp_path: Path) -> None:
    body = (
        "- id: onboarding-001\n  category: onboarding\n  title: One\n"
        "  critical: false\n  in_chat: true\n  steps:\n    - say: /help\n"
    )
    (tmp_path / "a.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(JourneySchemaError, match="duplicate"):
        load_journeys(tmp_path)


def test_help_journey_passes_and_a_missed_expect_fails(tmp_path: Path) -> None:
    passed = run_journey(parse_journey(_HELP), root=tmp_path / "ok")
    assert passed.passed, passed.failures
    missed = parse_journey({**_HELP, "steps": [{"say": "/help", "expect": ["NOT-IN-THE-OUTPUT"]}]})
    failed = run_journey(missed, root=tmp_path / "bad")
    assert not failed.passed
    assert any("missing" in item for item in failed.failures)


def test_planted_secret_in_a_config_file_is_caught(tmp_path: Path) -> None:
    journey = parse_journey(
        {
            **_HELP,
            "id": "onboarding-008",
            "in_chat": False,
            "steps": [
                {"say": "/help"},
                {"edit_config": {"path": "notes/leak.txt", "text": "{key_fx1}"}},
            ],
        }
    )
    result = run_journey(journey, root=tmp_path)
    assert not result.passed
    blob = "\n".join(result.failures)
    assert "planted secret" in blob
    assert "jrnk-" not in blob


def test_invalid_ledger_json_is_caught(tmp_path: Path) -> None:
    journey = parse_journey(
        {
            **_HELP,
            "id": "onboarding-007",
            "in_chat": False,
            "steps": [
                {"say": "/help"},
                {"edit_config": {"path": "jobs/bad.json", "text": "{"}},
            ],
        }
    )
    result = run_journey(journey, root=tmp_path)
    assert any("invalid JSON" in item for item in result.failures)


def test_display_invariants_catch_traceback_honesty_and_controls() -> None:
    traceback = InvariantView(
        chunks=("Traceback (most recent call last):",),
        logs="",
        blobs=(),
        json_errors=(),
        secrets=(),
        children=(),
        child_envs=(),
    )
    honesty = InvariantView(
        chunks=("The strategy Sharpe is 2.4 today.",),
        logs="",
        blobs=(),
        json_errors=(),
        secrets=(),
        children=(),
        child_envs=(),
    )
    controls = InvariantView(
        chunks=("hello \x1b[31mred",),
        logs="",
        blobs=(),
        json_errors=(),
        secrets=(),
        children=(),
        child_envs=(),
    )
    clean = InvariantView(
        chunks=("QLIKE is 0.12. SYNTHETIC correctness check only.",),
        logs="",
        blobs=(),
        json_errors=(),
        secrets=(),
        children=(),
        child_envs=(),
    )
    assert any("traceback" in item for item in invariant_problems(traceback))
    assert any("honesty" in item for item in invariant_problems(honesty))
    assert any("control" in item for item in invariant_problems(controls))
    assert invariant_problems(clean) == []


def test_orphan_detector_sees_a_live_child() -> None:
    before = child_pids()
    proc = subprocess.Popen(["sleep", "30"])  # noqa: S603 - fixed argv, reaped below
    try:
        assert proc.pid in unexpected_children(before)
    finally:
        proc.kill()
        proc.wait(timeout=2)


def test_down_endpoint_fails_visibly(tmp_path: Path) -> None:
    journey = parse_journey(
        {
            **_HELP,
            "id": "models-009",
            "category": "models",
            "setup": {"endpoints": {"fx1": "down"}},
            "steps": [{"say": "hello", "status": "error", "expect": ["model server failed"]}],
        }
    )
    result = run_journey(journey, root=tmp_path)
    assert result.passed, result.failures


def test_freeze_digest_changes_when_a_file_changes(tmp_path: Path) -> None:
    target = tmp_path / "a.yaml"
    target.write_text("id: onboarding-001\n", encoding="utf-8")
    write_freeze(tmp_path, tmp_path / "journeys.sha256")
    assert journey_digest(tmp_path) in (tmp_path / "journeys.sha256").read_text(encoding="utf-8")
    target.write_text("id: onboarding-002\n", encoding="utf-8")
    report = SuiteReport(
        results=(), digest=journey_digest(tmp_path), frozen=True, scope=SCOPE, elapsed_s=0
    )
    problems = gate_problems(report, tmp_path)
    assert any("digest" in item for item in problems)


def test_real_harness_mode_uses_the_lab_runner(tmp_path: Path) -> None:
    harness = make_harness(Setup(harness_mode="real"), tmp_path, [])
    assert type(harness.inner).__name__ == "Harness"


def _row(
    category: str, number: int, *, passed: bool = True, critical: bool = False
) -> JourneyResult:
    return JourneyResult(
        id=f"{category}-{number:03d}",
        category=category,
        title="gate row",
        critical=critical,
        in_chat=True,
        passed=passed,
        failures=() if passed else ("failed",),
        elapsed_s=0.01,
    )


def _hundred(*, fail: set[tuple[str, int]] | None = None) -> SuiteReport:
    failed = fail or set()
    rows = [
        _row(category, number, passed=(category, number) not in failed, critical=number == 1)
        for category in CATEGORIES
        for number in range(1, 11)
    ]
    return SuiteReport(results=tuple(rows), digest="abc", frozen=True, scope=SCOPE, elapsed_s=1)


def test_frozen_gate_allows_one_miss_and_rejects_a_critical_miss() -> None:
    assert evaluate(_hundred(), frozen=True) == []
    one_miss = evaluate(_hundred(fail={("export", 10)}), frozen=True)
    assert one_miss == []
    two_miss = evaluate(_hundred(fail={("export", 10), ("export", 9)}), frozen=True)
    assert any("in-chat" in item for item in two_miss)
    critical = evaluate(_hundred(fail={("export", 1)}), frozen=True)
    assert any("critical" in item for item in critical)


def test_unfrozen_gate_requires_every_loaded_journey(tmp_path: Path) -> None:
    green = SuiteReport(
        results=(_row("onboarding", 1),), digest="x", frozen=False, scope=SCOPE, elapsed_s=0
    )
    assert evaluate(green, frozen=False) == []
    red = SuiteReport(
        results=(_row("onboarding", 1, passed=False),),
        digest="x",
        frozen=False,
        scope=SCOPE,
        elapsed_s=0,
    )
    assert any("failures" in item for item in evaluate(red, frozen=False))
    assert not (tmp_path / "journeys.sha256").exists()
