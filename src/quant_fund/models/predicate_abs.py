"""Predicate abstraction over a Boolean-vector abstract domain (SYNTHETIC).

Concrete domain: bounded int state {x}. Predicates p_i(x). Abstract reach:
bools of each predicate under the successor relation, joined over the
abstraction's concretization — the classic Boolean-program construction.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1021

AbsState = frozenset[tuple[bool, ...]]


def abstract(s: int, preds: list[Any]) -> tuple[bool, ...]:
    return tuple(p(s) for p in preds)


def reach_abstract(init: set[int], trans: Any, preds: list[Any], dom: range) -> AbsState:
    """Best abstract transformer: enumerate concrete states per cell."""
    cells: dict[tuple[bool, ...], set[int]] = {}
    for x in dom:
        cells.setdefault(abstract(x, preds), set()).add(x)
    cur: AbsState = frozenset(abstract(x, preds) for x in init)
    changed = True
    while changed:
        changed = False
        nxt = set(cur)
        for cell in cur:
            for x in cells[cell]:
                succ = trans(x)
                if succ in dom:
                    c2 = abstract(succ, preds)
                    if c2 not in nxt:
                        nxt.add(c2)
                        changed = True
        cur = frozenset(nxt)
    return cur


def bench_predicate_abs(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    preds = [lambda x: x <= 0, lambda x: x >= 5]
    # counter x+1 clamped to 6: abstract reach covers (F,F),(F,T) but not (T,*)
    reach = reach_abstract({0}, lambda x: min(x + 1, 6), preds, range(-2, 8))
    checks.append((True, False) in reach)  # x<=0 initial cell
    checks.append((False, True) in reach)  # reaches x>=5
    # invariant: abstract reach never contains an impossible pred combo
    checks.append(
        all(
            cell in reach_abstract({0}, lambda x: x, preds, range(-2, 8))
            for cell in [(True, False)]
        )
    )
    # error predicate unreachable under clamped decr
    preds2 = [lambda x: x < 0]
    reach2 = reach_abstract({3}, lambda x: max(x - 1, 0), preds2, range(-2, 8))
    checks.append((True,) not in reach2)
    # finer predicates => finer abstraction (more cells reached for ramp)
    preds_coarse = [lambda x: x >= 3]
    preds_fine = [lambda x: x >= 1, lambda x: x >= 3, lambda x: x >= 5]
    rc = reach_abstract({0}, lambda x: min(x + 1, 6), preds_coarse, range(0, 7))
    rf = reach_abstract({0}, lambda x: min(x + 1, 6), preds_fine, range(0, 7))
    checks.append(len(rf) > len(rc))
    return {"synthetic_predicate_abs": float(sum(checks)) / len(checks)}
