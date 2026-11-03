#!/usr/bin/env python3
"""Line-pinned mutation campaign for live-trading safety paths.

mutmut and cosmic-ray are not locked dependencies. This runs the original
30 mutants plus about 40 more from ``scripts/mutation_test.py`` without
editing ``uv.lock`` or the working tree. Each pick is pinned by id, line,
and operator so a source move fails closed instead of scoring a different edit.

A mutation score is a test-suite diagnostic, not market evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import mutation_test as mt  # noqa: E402

TIMEOUT_S = 45.0


@dataclass(frozen=True)
class Pick:
    """One mutant, pinned so the id cannot silently point at a new edit."""

    mutant_id: str
    lineno: int
    operator: str


@dataclass(frozen=True)
class Suite:
    """One target file and the tests that cover it."""

    name: str
    target: str
    picks: tuple[Pick, ...]
    selectors: tuple[str, ...]


def _picks(rows: Sequence[tuple[str, int, str]]) -> tuple[Pick, ...]:
    return tuple(Pick(mutant_id, lineno, operator) for mutant_id, lineno, operator in rows)


SUITES: tuple[Suite, ...] = (
    Suite(
        name="kill-switch",
        target="src/quant_fund/monitoring/kill_switch.py",
        # M0000 and M0005 only rewrite ``if self.state not in VALID_STATES: pass``.
        # The body is empty, so those two edits are equivalent and stay in the score.
        picks=_picks(
            (
                ("M0000", 24, "negation-insertion"),
                ("M0001", 30, "negation-insertion"),
                ("M0003", 42, "negation-insertion"),
                ("M0004", 51, "negation-insertion"),
                ("M0005", 53, "negation-insertion"),
                ("M0006", 24, "comparison-swap"),
                ("M0007", 30, "comparison-swap"),
                ("M0009", 42, "comparison-swap"),
                ("M0010", 46, "comparison-swap"),
                ("M0011", 51, "comparison-swap"),
                ("M0012", 52, "const-bool-flip"),
                ("M0013", 54, "const-bool-flip"),
            )
        ),
        selectors=(
            "tests/property/test_accounting_invariants.py::test_kill_switch_blocks_every_order_until_human_reset",
            "tests/unit/backtest/test_kill_switch_extremes.py",
        ),
    ),
    Suite(
        name="pretrade-kernel",
        target="src/quant_fund/pretrade/kernel.py",
        picks=_picks(
            (
                ("M0164", 263, "comparison-swap"),
                ("M0263", 307, "comparison-swap"),
                ("M0264", 307, "comparison-swap"),
                ("M0217", 438, "comparison-swap"),
            )
        ),
        selectors=(
            "tests/property/test_accounting_invariants.py::test_approved_orders_stay_inside_hard_limits",
            "tests/unit/pretrade/test_kernel_boundaries.py",
        ),
    ),
    Suite(
        name="order-fill-ledger",
        target="src/quant_fund/execution/order_recon.py",
        # Duplicate order/fill membership became ``is not None`` after the
        # conflicting-replay merge. The comparison swap is that same gate.
        picks=_picks(
            (
                ("M0042", 250, "negation-insertion"),
                ("M0093", 250, "comparison-swap"),
                ("M0094", 254, "comparison-swap"),
                ("M0104", 327, "comparison-swap"),
            )
        ),
        selectors=(
            "tests/property/test_submit_idempotency_invariants.py",
            "tests/property/test_order_lifecycle_invariants.py",
            "tests/property/test_position_reconcile_invariants.py",
        ),
    ),
    Suite(
        name="paper-ledger",
        target="src/quant_fund/paper/ledger.py",
        picks=_picks(
            (
                ("M0055", 156, "numeric-perturb"),
                ("M0126", 211, "negation-insertion"),
                ("M0225", 211, "comparison-swap"),
                ("M0344", 223, "comparison-swap"),
            )
        ),
        selectors=(
            "tests/property/test_accounting_invariants.py::test_paper_ledger_conserves_cash_pnl_and_fees",
            "tests/property/test_accounting_invariants.py::test_fill_fees_are_monotone_in_quantity",
        ),
    ),
    Suite(
        name="broker-submit",
        target="src/quant_fund/execution/simulated_broker.py",
        picks=_picks(
            (
                ("M0073", 393, "negation-insertion"),
                ("M0074", 395, "negation-insertion"),
                ("M0077", 500, "negation-insertion"),
                ("M0159", 393, "comparison-swap"),
                ("M0160", 395, "comparison-swap"),
                ("M0173", 500, "comparison-swap"),
            )
        ),
        selectors=(
            "tests/regression/test_duplicate_order_submit.py",
            "tests/property/test_submit_idempotency_invariants.py",
        ),
    ),
    Suite(
        name="reconcile-drift",
        target="src/quant_fund/execution/order_recon.py",
        picks=_picks(
            (
                ("M0061", 418, "negation-insertion"),
                ("M0119", 418, "boolop-swap"),
                ("M0172", 418, "comparison-swap"),
                ("M0173", 418, "comparison-swap"),
                ("M0174", 428, "comparison-swap"),
                ("M0175", 428, "comparison-swap"),
                ("M0197", 429, "comparison-swap"),
                ("M0198", 429, "comparison-swap"),
            )
        ),
        selectors=(
            "tests/property/test_position_reconcile_invariants.py",
            "tests/unit/execution/test_reconcile_drift_boundaries.py",
        ),
    ),
    Suite(
        name="order-lifecycle",
        target="src/quant_fund/formal/order_lifecycle.py",
        picks=_picks(
            (
                ("M0005", 119, "negation-insertion"),
                ("M0042", 119, "comparison-swap"),
                ("M0043", 119, "comparison-swap"),
                ("M0094", 169, "comparison-swap"),
                ("M0058", 171, "negation-removal"),
                ("M0070", 215, "comparison-swap"),
                ("M0098", 217, "comparison-swap"),
                ("M0100", 222, "comparison-swap"),
            )
        ),
        selectors=(
            "tests/formal/test_order_lifecycle.py::test_hand_traces_for_races_restarts_and_duplicate_fills",
            "tests/formal/test_order_lifecycle.py::test_python_twin_enumerates_without_safety_violations",
            "tests/formal/test_lifecycle_boundaries.py",
        ),
    ),
    Suite(
        name="risk-ceilings",
        target="src/quant_fund/portfolio/risk_gate.py",
        # M0055 flips ``value > limit`` to ``>=`` on the early return. Equality
        # still returns False: a zero breach is not above the slack. It stays
        # in the score.
        picks=_picks(
            (
                ("M0003", 23, "negation-removal"),
                ("M0055", 23, "comparison-swap"),
                ("M0056", 23, "comparison-swap"),
                ("M0027", 25, "comparison-swap"),
                ("M0028", 25, "comparison-swap"),
                ("M0021", 128, "negation-insertion"),
                ("M0022", 130, "negation-insertion"),
                ("M0023", 132, "negation-insertion"),
            )
        ),
        selectors=(
            "tests/unit/risk/test_risk_gate.py",
            "tests/unit/risk/test_risk_gate_slack.py",
        ),
    ),
    Suite(
        name="daily-loss-floors",
        target="src/quant_fund/pretrade/engine.py",
        picks=_picks(
            (
                ("M0093", 417, "boolop-swap"),
                ("M0197", 417, "comparison-swap"),
                ("M0198", 417, "comparison-swap"),
                ("M0199", 417, "comparison-swap"),
            )
        ),
        selectors=(
            "tests/unit/pretrade/test_kernel_boundaries.py::test_nav_on_the_daily_floor_trips_and_just_above_does_not",
            "tests/unit/pretrade/test_engine.py::test_backward_session_roll_does_not_rebase_loss_floors",
            "tests/unit/pretrade/test_daily_loss_floor_boundaries.py",
        ),
    ),
    Suite(
        name="staleness",
        target="src/quant_fund/portfolio/risk_gate.py",
        picks=_picks(
            (
                ("M0049", 113, "negation-insertion"),
                ("M0073", 113, "comparison-swap"),
                ("M0074", 113, "comparison-swap"),
                ("M0050", 115, "negation-insertion"),
                ("M0075", 115, "comparison-swap"),
                ("M0076", 115, "comparison-swap"),
            )
        ),
        selectors=("tests/unit/risk/test_risk_gate.py",),
    ),
    Suite(
        name="live-broker",
        target="src/quant_fund/live/brokers.py",
        picks=_picks(
            (
                ("M0137", 371, "numeric-perturb"),
                ("M0201", 264, "comparison-swap"),
                ("M0202", 264, "comparison-swap"),
                ("M0204", 269, "comparison-swap"),
                ("M0205", 269, "comparison-swap"),
                ("M0095", 744, "comparison-swap"),
            )
        ),
        selectors=(
            "tests/unit/live/test_live_safety_branches.py::test_loopback_rejects_cash_position_and_side_without_raising",
            "tests/unit/live/test_prod_broker_errors.py",
            "tests/unit/live/test_broker_reject_boundaries.py",
        ),
    ),
)


def select_picks(mutants: Sequence[mt.Mutant], picks: Sequence[Pick]) -> list[mt.Mutant]:
    """Return the pinned mutants. A drifted id, line, or operator is an error."""
    by_id = {mutant.mutant_id: mutant for mutant in mutants}
    chosen: list[mt.Mutant] = []
    for pick in picks:
        mutant = by_id.get(pick.mutant_id)
        if mutant is None or mutant.lineno != pick.lineno or mutant.operator != pick.operator:
            found = "missing" if mutant is None else f"line {mutant.lineno} {mutant.operator}"
            raise mt.MutationTestError(
                f"{pick.mutant_id} expected line {pick.lineno} {pick.operator}, found {found}"
            )
        chosen.append(mutant)
    return chosen


def stable_pytest_command(selectors: Sequence[str]) -> list[str]:
    """The harness command plus a fixed plugin set, so order does not move."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-x",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
        *selectors,
    ]
    return command


def run_suite(
    suite: Suite,
    *,
    repo: Path = REPO,
    timeout_s: float = TIMEOUT_S,
    runner: mt.Runner | None = None,
) -> mt.CampaignReport:
    """Execute one suite. The target file on disk is not modified."""
    selectors = mt.validate_selectors(suite.selectors, repo)
    target = mt.resolve_target(suite.target, repo)
    before = target.read_bytes()
    module_name = mt.module_name_for(target)
    source = target.read_text(encoding="utf-8")
    mutants = mt.generate_mutants(source, filename=str(target))
    chosen = select_picks(mutants, suite.picks)
    previous_command = mt.build_command
    mt.build_command = stable_pytest_command  # type: ignore[method-assign]
    try:
        with tempfile.TemporaryDirectory(prefix=f"mutation-{suite.name}-") as tmp:
            results = mt.run_campaign(
                chosen,
                selectors,
                repo=repo,
                overlay_dir=Path(tmp) / "overlay",
                module_name=module_name,
                timeout_s=timeout_s,
                runner=runner,
                original_source=source,
            )
    finally:
        mt.build_command = previous_command  # type: ignore[method-assign]
    if target.read_bytes() != before:
        raise mt.MutationTestError(f"{target} changed during the campaign")
    return mt.build_report(
        target,
        module_name,
        selectors,
        seed=0,
        timeout_s=timeout_s,
        total_mutants=len(mutants),
        results=results,
    )


def totals(reports: Sequence[mt.CampaignReport]) -> dict[str, Any]:
    """Score across suites. Errors stay out of the denominator."""
    counts = {"killed": 0, "survived": 0, "timeout": 0, "error": 0}
    for report in reports:
        for key, value in report.counts.items():
            counts[key] += int(value)
    attempted = sum(counts.values()) - counts["error"]
    score = (counts["killed"] + counts["timeout"]) / attempted if attempted else 0.0
    return {"counts": counts, "sampled": sum(counts.values()), "score": round(score, 4)}


def campaign_payload(reports: Sequence[mt.CampaignReport]) -> dict[str, Any]:
    """JSON body for one before or after run."""
    return {
        "watermark": mt.WATERMARK,
        "timeout_s": reports[0].timeout_s if reports else TIMEOUT_S,
        "suites": [mt.report_to_dict(report) for report in reports],
        "totals": totals(reports),
    }


def format_campaign(payload: dict[str, Any]) -> str:
    """Plain-text scoreboard."""
    lines: list[str] = []
    for suite in payload["suites"]:
        counts = suite["counts"]
        lines.append(
            f"{Path(suite['target']).name}: score {suite['score']:.2%} "
            f"killed={counts['killed']} survived={counts['survived']} "
            f"timeout={counts['timeout']} error={counts['error']} "
            f"sampled={suite['sampled']}/{suite['total_mutants']}"
        )
        for row in suite["survivors"]:
            lines.append(
                f"  survived {row['mutant_id']} line {row['lineno']}: "
                f"[{row['operator']}] {row['description']}"
            )
    overall = payload["totals"]
    counts = overall["counts"]
    lines.append(
        f"overall: score {overall['score']:.2%} "
        f"killed={counts['killed']} survived={counts['survived']} "
        f"timeout={counts['timeout']} error={counts['error']} sampled={overall['sampled']}"
    )
    lines.append(mt.WATERMARK)
    return "\n".join(lines)


def below_min_score(score: float, minimum: float | None) -> bool:
    """True when a threshold is set and the score is under it."""
    return minimum is not None and score < minimum


def suites_by_name(names: Sequence[str] | None) -> tuple[Suite, ...]:
    """Resolve ``--only`` names. Empty means every suite."""
    if not names:
        return SUITES
    by_name = {suite.name: suite for suite in SUITES}
    missing = [name for name in names if name not in by_name]
    if missing:
        known = ", ".join(suite.name for suite in SUITES)
        raise mt.MutationTestError(f"unknown suite {missing}; known: {known}")
    return tuple(by_name[name] for name in names)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="suite name to run; repeatable. Default is every suite.",
    )
    parser.add_argument("--timeout", type=float, default=TIMEOUT_S)
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="fail when the overall score is below this threshold in [0, 1]",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        raise mt.MutationTestError(f"--timeout must be > 0, got {args.timeout}")
    if args.min_score is not None and not 0.0 <= args.min_score <= 1.0:
        raise mt.MutationTestError(f"--min-score must be in [0, 1], got {args.min_score}")
    selected = suites_by_name(args.only)
    reports: list[mt.CampaignReport] = []
    for suite in selected:
        reports.append(run_suite(suite, timeout_s=args.timeout))
        print(mt.format_summary(reports[-1]), flush=True)
        print("", flush=True)
    payload = campaign_payload(reports)
    text = format_campaign(payload)
    print(text, flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    counts = payload["totals"]["counts"]
    failed = bool(counts["error"] or counts["timeout"])
    if below_min_score(float(payload["totals"]["score"]), args.min_score):
        print(f"min-score gate: FAIL (>= {args.min_score:.2%})", flush=True)
        failed = True
    elif args.min_score is not None:
        print(f"min-score gate: PASS (>= {args.min_score:.2%})", flush=True)
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except mt.MutationTestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
