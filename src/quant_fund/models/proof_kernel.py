"""Natural-deduction proof kernel for propositional logic: sequents
ctx |- formula with intro/elim rules only through the kernel
(SYNTHETIC bench only)."""

from __future__ import annotations

from dataclasses import dataclass

_SEED = 20261231 + 997


@dataclass(frozen=True)
class Form:
    tag: str
    a: Form | str | None = None
    b: Form | str | None = None


def ATOM(name: str) -> Form:
    return Form("atom", name)


def IMP(a: Form, b: Form) -> Form:
    return Form("->", a, b)


def AND(a: Form, b: Form) -> Form:
    return Form("&", a, b)


def OR(a: Form, b: Form) -> Form:
    return Form("|", a, b)


BOT = Form("bot")


@dataclass(frozen=True)
class Sequent:
    ctx: frozenset
    concl: Form


def assume(f: Form) -> Sequent:
    return Sequent(frozenset({f}), f)


def weaken(sq: Sequent, hyp: Form) -> Sequent:
    return Sequent(sq.ctx | frozenset({hyp}), sq.concl)


def imp_intro(sq: Sequent, hyp: Form) -> Sequent:
    ctx = set(sq.ctx)
    if hyp not in ctx:
        raise ValueError("hyp not in context")
    ctx.discard(hyp)
    return Sequent(frozenset(ctx), IMP(hyp, sq.concl))


def imp_elim(s1: Sequent, s2: Sequent) -> Sequent:
    if s1.concl.tag != "->":
        raise ValueError("not implication")
    if s2.concl != s1.concl.a:
        raise ValueError("antecedent mismatch")
    return Sequent(s1.ctx | s2.ctx, s1.concl.b)  # type: ignore[arg-type]


def and_intro(s1: Sequent, s2: Sequent) -> Sequent:
    return Sequent(s1.ctx | s2.ctx, AND(s1.concl, s2.concl))


def and_elim_l(s: Sequent) -> Sequent:
    if s.concl.tag != "&":
        raise ValueError("not conjunction")
    return Sequent(s.ctx, s.concl.a)  # type: ignore[arg-type]


def and_elim_r(s: Sequent) -> Sequent:
    if s.concl.tag != "&":
        raise ValueError("not conjunction")
    return Sequent(s.ctx, s.concl.b)  # type: ignore[arg-type]


def or_intro_l(s: Sequent, other: Form) -> Sequent:
    return Sequent(s.ctx, OR(s.concl, other))


def or_elim(s_or: Sequent, s1: Sequent, s2: Sequent) -> Sequent:
    if s_or.concl.tag != "|":
        raise ValueError("not disjunction")
    a, b = s_or.concl.a, s_or.concl.b
    if a not in s1.ctx or b not in s2.ctx or s1.concl != s2.concl:
        raise ValueError("cases mismatch")
    c1 = set(s1.ctx)
    c1.discard(a)
    c2 = set(s2.ctx)
    c2.discard(b)
    return Sequent(s_or.ctx | frozenset(c1) | frozenset(c2), s1.concl)


def bench_proof_kernel(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    a, b = ATOM("A"), ATOM("B")
    inner = weaken(assume(a), b)
    k = imp_intro(imp_intro(inner, b), a)
    checks.append(k.concl == IMP(a, IMP(b, a)) and len(k.ctx) == 0)
    checks.append(imp_intro(assume(a), a).concl == IMP(a, a))
    ab = assume(IMP(a, b))
    av = assume(a)
    bv = imp_elim(ab, av)
    checks.append(bv.concl == b and bv.ctx == frozenset({IMP(a, b), a}))
    conj = and_intro(assume(a), assume(b))
    checks.append(and_elim_l(conj).concl == a and and_elim_r(conj).concl == b)
    try:
        imp_elim(assume(a), assume(b))
        checks.append(False)
    except ValueError:
        checks.append(True)
    try:
        imp_intro(assume(a), b)
        checks.append(False)
    except ValueError:
        checks.append(True)
    swap = imp_intro(
        and_intro(and_elim_r(assume(AND(a, b))), and_elim_l(assume(AND(a, b)))),
        AND(a, b),
    )
    checks.append(swap.concl == IMP(AND(a, b), AND(b, a)) and len(swap.ctx) == 0)
    return {"synthetic_proof_kernel": float(sum(checks) / len(checks))}
