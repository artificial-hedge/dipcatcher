"""Counterexample-guided inductive synthesis (CEGIS) over a stratified (SYNTHETIC)
LIA grammar — same term/guard stratification as the ESolver.

Loop: synthesizer proposes the first candidate (ordered small-first)
consistent with all counterexamples; verifier checks the spec on the
bounded domain and either accepts or returns one counterexample.
"""

from __future__ import annotations

from typing import Any

from quant_fund.models.sygus_synth import _eval

_SEED = 20261231 + 1022


def _lib(vars_: list[str]) -> list[Any]:
    """Ordered candidate library: terms, then ite(guard, term, term)."""
    atoms: list[Any] = [("lit", 0), ("lit", 1)] + [("var", v) for v in vars_]
    terms = list(atoms)
    for a in atoms:
        for b in atoms:
            terms.append(("add", a, b))
            terms.append(("sub", a, b))
    guards = [("le", a, b) for a in atoms for b in atoms]
    out = list(terms)
    for g in guards:
        for a in terms:
            for b in terms:
                out.append(("ite", g, a, b))
    return out


def cegis(spec_fn: Any, vars_: list[str], dom: range) -> tuple[Any | None, int]:
    import itertools

    dom_envs = [
        dict(zip(vars_, vals, strict=True)) for vals in itertools.product(dom, repeat=len(vars_))
    ]
    cexs: list[dict[str, int]] = []
    cands = _lib(vars_)
    iters = 0
    while iters < 60:
        iters += 1
        cand = None
        for t in cands:
            if all(spec_fn(_eval(t, io), io) for io in cexs):
                cand = t
                break
        if cand is None:
            return None, iters
        bad = None
        for env in dom_envs:
            if not spec_fn(_eval(cand, env), env):
                bad = env
                break
        if bad is None:
            return cand, iters
        cexs.append(bad)
        # spec-guided prune: drop every candidate that violates the spec on
        # this env (not just those matching the recorded wrong output)
        cands = [t for t in cands if spec_fn(_eval(t, bad), bad)]
    return None, iters


def bench_cegis_loop(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # synth max via CEGIS — spec out>=x,out>=y,out in {x,y}
    spec = lambda out, e: out >= e["x"] and out >= e["y"] and out in (e["x"], e["y"])  # noqa: E731
    t, it = cegis(spec, ["x", "y"], range(-2, 3))
    checks.append(t is not None and it <= 12)
    checks.append(_eval(t, {"x": 9, "y": -9}) == 9 if t else False)
    # synth clamp-lo: f(x) = max(x,0); spec: out>=0 ∧ out>=x ∧ (out==0 ∨ out==x)
    spec2 = lambda out, e: out >= 0 and out >= e["x"] and out in (0, e["x"])  # noqa: E731
    t2, it2 = cegis(spec2, ["x"], range(-3, 4))
    checks.append(t2 is not None and it2 <= 12)
    checks.append(_eval(t2, {"x": -4}) == 0 if t2 else False)
    # unsatisfiable spec returns None quickly (out = x+100 unbuildable)
    spec3 = lambda out, e: out == e["x"] + 100  # noqa: E731
    t3, _ = cegis(spec3, ["x"], range(-2, 3))
    checks.append(t3 is None)
    return {"synthetic_cegis_loop": float(sum(checks)) / len(checks)}
