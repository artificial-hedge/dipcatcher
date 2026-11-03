"""The line-pinned campaign selects stable mutants and leaves the tree alone."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts import mutation_test as mt
from scripts.mutation_campaign import (
    SUITES,
    below_min_score,
    select_picks,
    stable_pytest_command,
    suites_by_name,
)
from scripts.mutation_campaign import (
    Pick as CampaignPick,
)
from scripts.mutation_campaign import (
    run_suite as run_campaign_suite,
)


def test_curated_campaign_pins_seventy_mutants() -> None:
    picks = [(suite.target, pick.mutant_id) for suite in SUITES for pick in suite.picks]
    assert len(picks) == 70
    assert len(set(picks)) == 70
    names = [suite.name for suite in SUITES]
    assert names == [
        "kill-switch",
        "pretrade-kernel",
        "order-fill-ledger",
        "paper-ledger",
        "broker-submit",
        "reconcile-drift",
        "order-lifecycle",
        "risk-ceilings",
        "daily-loss-floors",
        "staleness",
        "live-broker",
    ]
    added = {
        "reconcile-drift",
        "order-lifecycle",
        "risk-ceilings",
        "daily-loss-floors",
        "staleness",
        "live-broker",
    }
    assert sum(len(suite.picks) for suite in SUITES if suite.name in added) == 40


def test_mutation_make_target_fails_under_ninety_percent() -> None:
    recipe = Path("deploy/mutation.mk").read_text(encoding="utf-8")
    assert "mutation-campaign:" in recipe
    assert "--min-score 0.90" in recipe
    assert "deploy/mutation.mk" not in Path("Makefile").read_text(encoding="utf-8")


def test_min_score_threshold_is_strict() -> None:
    assert below_min_score(0.8999, 0.90)
    assert not below_min_score(0.90, 0.90)
    assert not below_min_score(0.80, None)


def test_select_picks_rejects_a_drifted_operator() -> None:
    mutants = mt.generate_mutants("if a <= b:\n    pass\n", filename="fixture.py")
    comparison = next(mutant for mutant in mutants if mutant.operator == "comparison-swap")
    chosen = select_picks(
        mutants,
        [CampaignPick(comparison.mutant_id, comparison.lineno, comparison.operator)],
    )
    assert chosen == [comparison]
    with pytest.raises(mt.MutationTestError, match="expected"):
        select_picks(
            mutants,
            [CampaignPick(comparison.mutant_id, comparison.lineno, "numeric-perturb")],
        )


def test_unknown_suite_name_is_rejected() -> None:
    with pytest.raises(mt.MutationTestError, match="unknown suite"):
        suites_by_name(["nope"])


def test_stable_command_disables_random_order() -> None:
    command = stable_pytest_command(("tests/property/test_accounting_invariants.py",))
    assert "-p" in command and "no:randomly" in command
    assert command[-1].endswith("test_accounting_invariants.py")


def test_kill_switch_suite_restores_the_harness_and_the_file() -> None:
    suite = next(item for item in SUITES if item.name == "kill-switch")
    target = Path(suite.target)
    if not target.is_absolute():
        target = Path(mt.REPO) / target
    before = target.read_bytes()
    original_command = mt.build_command
    calls: list[list[str]] = []

    def runner(command, cwd, env, timeout_s):
        del cwd, timeout_s
        calls.append(command)
        content = Path(env["MUTATION_TARGET_FILE"]).read_text(encoding="utf-8")
        if content == before.decode("utf-8") or mt.SENTINEL_MARKER in content:
            return mt.RunOutcome(
                returncode=0 if mt.SENTINEL_MARKER not in content else 1, duration_s=0.0
            )
        return mt.RunOutcome(returncode=1, duration_s=0.0)

    report = run_campaign_suite(suite, runner=runner, timeout_s=5)
    assert mt.build_command is original_command
    assert target.read_bytes() == before
    assert report.sampled == 12
    assert report.counts["killed"] == 12
    assert report.score == 1.0
    assert calls
    assert "no:randomly" in calls[0]
