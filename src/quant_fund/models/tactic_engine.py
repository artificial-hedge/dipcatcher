"""Mini tactic engine over proof_kernel goals: intro/split/assumption/
exact/apply tactics assembling kernel-checked Sequents
(SYNTHETIC bench only)."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_fund.models.proof_kernel import (
    AND,
    ATOM,
    IMP,
    Form,
    Sequent,
    and_intro,
    assume,
    imp_elim,
    imp_intro,
    or_intro_l,
    weaken,
)

_SEED = 20261231 + 998


class TacticError(Exception):
    pass


@dataclass
class Goal:
    ctx: frozenset
    concl: Form
    children: list[Goal] = field(default_factory=list)
    tactic: str = ""
    arg: object = None


def run(goal: Sequent, script: list[tuple]) -> Sequent:
    open_goals = [Goal(goal.ctx, goal.concl)]
    for tac in script:
        if not open_goals:
            raise TacticError("script overruns goals")
        g = open_goals.pop(0)
        g.tactic = tac[0]
        op = tac[0]
        if op == "intro":
            if g.concl.tag != "->":
                raise TacticError("intro on non-implication")
            g.arg = g.concl.a
            g.children = [Goal(g.ctx | frozenset({g.concl.a}), g.concl.b)]  # type: ignore[list-item,arg-type]
        elif op == "split":
            if g.concl.tag != "&":
                raise TacticError("split on non-conjunction")
            g.children = [
                Goal(g.ctx, g.concl.a),  # type: ignore[list-item,arg-type]
                Goal(g.ctx, g.concl.b),  # type: ignore[list-item,arg-type]
            ]
        elif op == "left":
            if g.concl.tag != "|":
                raise TacticError("left on non-disjunction")
            g.arg = g.concl.b
            g.children = [Goal(g.ctx, g.concl.a)]  # type: ignore[list-item,arg-type]
        elif op == "assumption":
            if g.concl not in g.ctx:
                raise TacticError("no matching hypothesis")
        elif op == "exact":
            thm = tac[1]
            if not isinstance(thm, Sequent) or thm.concl != g.concl:
                raise TacticError("exact mismatch")
            if not thm.ctx.issubset(g.ctx):
                raise TacticError("exact uses unproven premises")
            g.arg = thm
        elif op == "apply":
            thm = tac[1]
            if not isinstance(thm, Sequent) or thm.concl.tag != "->":
                raise TacticError("apply needs implication")
            if thm.concl.b != g.concl:
                raise TacticError("apply conclusion mismatch")
            if not thm.ctx.issubset(g.ctx):
                raise TacticError("apply uses unproven premises")
            g.arg = thm
            g.children = [Goal(g.ctx, thm.concl.a)]  # type: ignore[list-item,arg-type]
        else:
            raise TacticError(f"unknown tactic {op}")
        open_goals = g.children + open_goals
    if open_goals:
        raise TacticError("unproven goals remain")
    return _prove(_root_from(goal, script))


def _root_from(goal: Sequent, script: list[tuple]) -> Goal:
    # Re-run the script to materialize the goal tree (kept small).
    root = Goal(goal.ctx, goal.concl)
    open_goals = [root]
    for tac in script:
        g = open_goals.pop(0)
        g.tactic = tac[0]
        op = tac[0]
        if op == "intro":
            g.arg = g.concl.a
            g.children = [Goal(g.ctx | frozenset({g.concl.a}), g.concl.b)]  # type: ignore[list-item,arg-type]
        elif op == "split":
            g.children = [
                Goal(g.ctx, g.concl.a),  # type: ignore[list-item,arg-type]
                Goal(g.ctx, g.concl.b),  # type: ignore[list-item,arg-type]
            ]
        elif op == "left":
            g.arg = g.concl.b
            g.children = [Goal(g.ctx, g.concl.a)]  # type: ignore[list-item,arg-type]
        elif op == "exact":
            g.arg = tac[1]
        elif op == "apply":
            g.arg = tac[1]
            g.children = [Goal(g.ctx, tac[1].concl.a)]
        open_goals = g.children + open_goals
    return root


def _prove(g: Goal) -> Sequent:
    op = g.tactic
    if op == "intro":
        child = _prove(g.children[0])
        return imp_intro(child, g.arg)  # type: ignore[arg-type]
    if op == "split":
        return and_intro(_prove(g.children[0]), _prove(g.children[1]))
    if op == "left":
        return or_intro_l(_prove(g.children[0]), g.arg)  # type: ignore[arg-type]
    if op == "assumption":
        sq = assume(g.concl)
        for h in g.ctx - frozenset({g.concl}):
            sq = weaken(sq, h)
        return sq
    if op == "exact":
        thm = g.arg
        if not isinstance(thm, Sequent):
            raise TacticError("exact arg not a sequent")
        sq = thm
        for h in g.ctx - thm.ctx:
            sq = weaken(sq, h)
        return sq
    if op == "apply":
        return imp_elim(g.arg, _prove(g.children[0]))  # type: ignore[arg-type]
    raise TacticError("unproved goal")


def bench_tactic_engine(seed: int = _SEED) -> dict[str, float]:
    del seed
    a, b = ATOM("A"), ATOM("B")
    checks: list[bool] = []
    sq = run(Sequent(frozenset(), IMP(a, a)), [("intro",), ("assumption",)])
    checks.append(sq.concl == IMP(a, a) and len(sq.ctx) == 0)
    sq2 = run(
        Sequent(frozenset(), IMP(AND(a, b), AND(b, a))),
        [
            ("intro",),
            ("split",),
            ("exact", Sequent(frozenset({AND(a, b)}), b)),
            ("exact", Sequent(frozenset({AND(a, b)}), a)),
        ],
    )
    checks.append(sq2.concl == IMP(AND(a, b), AND(b, a)))
    # apply: prove B from A->B hypothesis
    ab = Sequent(frozenset({IMP(a, b), a}), b)
    sq3 = run(ab, [("apply", assume(IMP(a, b))), ("assumption",)])
    checks.append(sq3.concl == b)
    try:
        run(Sequent(frozenset(), a), [("assumption",)])
        checks.append(False)
    except TacticError:
        checks.append(True)
    try:
        run(Sequent(frozenset(), IMP(a, a)), [("assumption",)])
        checks.append(False)
    except TacticError:
        checks.append(True)
    try:
        run(Sequent(frozenset(), IMP(a, a)), [("intro",)])
        checks.append(False)
    except TacticError:
        checks.append(True)
    return {"synthetic_tactic_engine": float(sum(checks) / len(checks))}
