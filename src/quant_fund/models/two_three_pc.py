"""SYNTHETIC 2PC/3PC atomic commitment.

2PC: coordinator collects votes → unanimous commit else abort; verify
agreement (all participants reach the same decision) and validity
(commit only if all voted yes, abort if any no/crash).
3PC: pre-commit phase separates "committable" state; under participant
failure a live cohort can finish independently — verify termination.
"""

from __future__ import annotations

import random


def run_2pc(votes: list[bool], crash_before_commit: bool) -> list[str]:
    if crash_before_commit or not all(votes):
        return ["abort"] * len(votes)
    return ["commit"] * len(votes)


def run_3pc(votes: list[bool], coordinator_alive: bool) -> list[str]:
    if not all(votes):
        return ["abort"] * len(votes)
    # pre-commit reached; cohorts commit even if coordinator dies
    _ = coordinator_alive
    return ["commit"] * len(votes)


def bench_two_three_pc(seed: int = 20261231 + 445) -> dict[str, float]:
    rng = random.Random(seed)
    agree = valid = term = 0
    trials = 40
    for _ in range(trials):
        n = rng.randrange(2, 6)
        votes = [rng.random() < 0.75 for _ in range(n)]
        crash = rng.random() < 0.3
        d2 = run_2pc(votes, crash)
        agree += int(len(set(d2)) == 1)
        # validity: commit iff all yes and no crash; abort otherwise
        expect = "commit" if (all(votes) and not crash) else "abort"
        valid += int(set(d2) == {expect})
        # 3PC terminates (commit path taken under all-yes regardless)
        d3 = run_3pc(votes, coordinator_alive=not crash)
        term += int(len(set(d3)) == 1 and set(d3) <= {"commit", "abort"})
    return {
        "synthetic_agreement": float(agree / trials),
        "synthetic_validity": float(valid / trials),
        "synthetic_termination": float(term / trials),
    }
