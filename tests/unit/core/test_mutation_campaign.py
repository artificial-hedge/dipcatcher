"""The line-pinned campaign selects stable mutants and leaves the tree alone."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts import mutation_test as mt
from scripts.mutation_campaign import (
    EQUIVALENT,
    SUITES,
    below_min_score,
    reclassify_equivalent,
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


def test_curated_campaign_pins_every_suite() -> None:
    picks = [(suite.target, pick.mutant_id) for suite in SUITES for pick in suite.picks]
    assert len(picks) == 87
    assert len(set(picks)) == 87
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
        "live-risk-limits",
        "live-market-data",
    ]
    assert sum(len(suite.picks) for suite in SUITES if suite.name == "live-risk-limits") == 8
    assert sum(len(suite.picks) for suite in SUITES if suite.name == "live-market-data") == 9


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


def test_equivalent_survivors_leave_the_score_and_kills_stay_kills() -> None:
    by_name = {suite.name: suite for suite in SUITES}
    for item in EQUIVALENT:
        picks = [pick for pick in by_name[item.suite].picks if pick.mutant_id == item.mutant_id]
        assert picks == [CampaignPick(item.mutant_id, item.lineno, item.operator)]
    survived = mt.CampaignReport(
        created_utc="t",
        target="kill_switch.py",
        module="quant_fund.monitoring.kill_switch",
        selectors=(),
        seed=0,
        timeout_s=1.0,
        total_mutants=2,
        sampled=2,
        counts={"killed": 0, "survived": 2, "timeout": 0, "error": 0},
        score=0.0,
        results=[
            {
                "mutant_id": "M0000",
                "lineno": 23,
                "operator": "negation-insertion",
                "status": "survived",
                "description": "empty pass",
            },
            {
                "mutant_id": "M0001",
                "lineno": 29,
                "operator": "negation-insertion",
                "status": "survived",
                "description": "real branch",
            },
        ],
        survivors=[],
    )
    reclassify_equivalent(survived, "kill-switch")
    assert survived.results[0]["status"] == "equivalent"
    assert survived.results[1]["status"] == "survived"
    assert survived.counts["equivalent"] == 1
    assert survived.counts["survived"] == 1
    assert survived.counts["killed"] == 0
    assert survived.score == 0.0
    killed = mt.CampaignReport(
        created_utc="t",
        target="kill_switch.py",
        module="quant_fund.monitoring.kill_switch",
        selectors=(),
        seed=0,
        timeout_s=1.0,
        total_mutants=1,
        sampled=1,
        counts={"killed": 1, "survived": 0, "timeout": 0, "error": 0},
        score=1.0,
        results=[
            {
                "mutant_id": "M0000",
                "lineno": 23,
                "operator": "negation-insertion",
                "status": "killed",
                "description": "empty pass",
            }
        ],
        survivors=[],
    )
    reclassify_equivalent(killed, "kill-switch")
    assert killed.counts["killed"] == 1
    assert killed.score == 1.0
    assert "equivalent" not in killed.counts


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
