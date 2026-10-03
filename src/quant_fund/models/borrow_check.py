"""NLL-lite borrow checker over a statement-level IR.

IR: list of ops — ("borrow",x,k,loan_id) creates &k x loan, ("use",x),
("mut",x) mutation, ("die",x) end of x's region (last use implicit).
A shared borrow conflicts with later mutation while the loan is live
(loan live until its last use — NLL's key insight: regions are sets of
program points, not lexical scopes). Mut borrows conflict with any
overlapping loan or use of the same place.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1035

Op = tuple


def _loan_uses(ops: list[Op], loan: Any) -> list[int]:
    return [i for i, op in enumerate(ops) if op[0] == "loan_use" and op[1] == loan]


def check(ops: list[Op]) -> list[str]:
    """Return list of borrow errors (empty = clean)."""
    errors: list[str] = []
    live: list[tuple[Any, str, Any]] = []  # (place, kind, loan_id->tag)
    loans: dict[Any, tuple[Any, str]] = {}  # loan_id -> (place, kind)
    for i, op in enumerate(ops):
        tag = op[0]
        if tag == "borrow":
            _, x, kind, loan_id = op
            # check conflict with still-live earlier loans on x
            for p, k2, lid2 in live:
                if p == x and not (kind == "shared" and k2 == "shared"):
                    errors.append(f"borrow {loan_id} conflicts with live {lid2} at {i}")
            loans[loan_id] = (x, kind)
            live.append((x, kind, loan_id))
        elif tag == "loan_use":
            pass
        elif tag == "mut":
            _, x = op
            for p, k2, lid2 in live:
                if p == x:
                    errors.append(f"mutation of {x} at {i} while {k2} loan {lid2} live")
        elif tag == "use":
            pass
        elif tag == "die":
            _, x = op
            live = [(p, k, lid) for p, k, lid in live if p != x]
        # expire loans whose last use has passed
        still: list[tuple[Any, str, Any]] = []
        for p, k, lid in live:
            uses = _loan_uses(ops[i + 1 :], lid)
            if uses:
                still.append((p, k, lid))
        live = still
    return errors


def bench_borrow_check(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # NLL acceptance: shared loan ends at last use — mut after last use is OK
    ok_ops = [
        ("borrow", "x", "shared", "L1"),
        ("loan_use", "L1"),
        ("mut", "x"),
    ]
    checks.append(check(ok_ops) == [])
    # mut while shared loan still live (use comes after) -> error
    bad_ops = [
        ("borrow", "x", "shared", "L1"),
        ("mut", "x"),
        ("loan_use", "L1"),
    ]
    errs = check(bad_ops)
    checks.append(len(errs) == 1 and "mutation" in errs[0])
    # two shared loans coexist fine
    checks.append(
        check(
            [
                ("borrow", "x", "shared", "L1"),
                ("borrow", "x", "shared", "L2"),
                ("loan_use", "L1"),
                ("loan_use", "L2"),
            ]
        )
        == []
    )
    # mut loan conflicts with live shared loan
    checks.append(
        len(
            check(
                [
                    ("borrow", "x", "shared", "L1"),
                    ("borrow", "x", "mut", "L2"),
                    ("loan_use", "L1"),
                    ("loan_use", "L2"),
                ]
            )
        )
        == 1
    )
    # two mut loans conflict
    checks.append(
        len(
            check(
                [
                    ("borrow", "x", "mut", "L1"),
                    ("borrow", "x", "mut", "L2"),
                    ("loan_use", "L1"),
                    ("loan_use", "L2"),
                ]
            )
        )
        == 1
    )
    # sequential disjoint mut loans OK
    checks.append(
        check(
            [
                ("borrow", "x", "mut", "L1"),
                ("loan_use", "L1"),
                ("borrow", "x", "mut", "L2"),
                ("loan_use", "L2"),
            ]
        )
        == []
    )
    return {"synthetic_borrow_check": float(sum(checks)) / len(checks)}
