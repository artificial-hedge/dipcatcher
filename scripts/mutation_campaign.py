#!/usr/bin/env python3
"""Line-pinned mutation campaign for the ledger, kernel, kill switch, and broker.

mutmut and cosmic-ray are not locked dependencies. This runs about 30
single-edit mutants from ``scripts/mutation_test.py`` without editing
``uv.lock`` or the working tree. Each pick is pinned by id, line, and
operator so a source move fails closed instead of scoring a different edit.

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
                ("M0000", 23, "negation-insertion"),
                ("M0001", 29, "negation-insertion"),
                ("M0002", 35, "negation-insertion"),
                ("M0003", 42, "negation-insertion"),
                ("M0004", 44, "negation-insertion"),
                ("M0005", 23, "comparison-swap"),
                ("M0006", 29, "comparison-swap"),
                ("M0007", 35, "comparison-swap"),
                ("M0008", 39, "comparison-swap"),
                ("M0009", 42, "comparison-swap"),
                ("M0010", 43, "const-bool-flip"),
                ("M0011", 45, "const-bool-flip"),
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
                ("M0161", 237, "comparison-swap"),
                ("M0190", 280, "comparison-swap"),
                ("M0191", 280, "comparison-swap"),
                ("M0217", 386, "comparison-swap"),
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
        picks=_picks(
            (
                ("M0015", 171, "negation-insertion"),
                ("M0046", 171, "comparison-swap"),
                ("M0047", 173, "comparison-swap"),
                ("M0055", 212, "comparison-swap"),
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
                ("M0052", 147, "numeric-perturb"),
                ("M0123", 197, "negation-insertion"),
                ("M0222", 197, "comparison-swap"),
                ("M0342", 209, "comparison-swap"),
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
                ("M0023", 167, "negation-insertion"),
                ("M0024", 169, "negation-insertion"),
                ("M0027", 274, "negation-insertion"),
                ("M0072", 167, "comparison-swap"),
                ("M0073", 169, "comparison-swap"),
                ("M0086", 274, "comparison-swap"),
            )
        ),
        selectors=(
            "tests/regression/test_duplicate_order_submit.py",
            "tests/property/test_submit_idempotency_invariants.py",
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
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        raise mt.MutationTestError(f"--timeout must be > 0, got {args.timeout}")
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
    if counts["error"] or counts["timeout"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except mt.MutationTestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
