"""Sign-domain abstract interpretation: {-, 0, +, ⊤} lattice over exprs (SYNTHETIC).

Transfer functions follow the standard sign multiplication/addition tables;
soundness verified by concretizing against sampled concrete evaluations.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 1006

NEG, ZERO, POS, TOP = "-", "0", "+", "T"


def _lub(a: str, b: str) -> str:
    if a == b:
        return a
    if a == "0":
        return b
    if b == "0":
        return a
    return TOP


_MUL = {
    (NEG, NEG): POS,
    (NEG, ZERO): ZERO,
    (NEG, POS): NEG,
    (NEG, TOP): TOP,
    (ZERO, NEG): ZERO,
    (ZERO, ZERO): ZERO,
    (ZERO, POS): ZERO,
    (ZERO, TOP): ZERO,
    (POS, NEG): NEG,
    (POS, ZERO): ZERO,
    (POS, POS): POS,
    (POS, TOP): TOP,
    (TOP, NEG): TOP,
    (TOP, ZERO): ZERO,
    (TOP, POS): TOP,
    (TOP, TOP): TOP,
}

_ADD = {
    (NEG, NEG): NEG,
    (NEG, ZERO): NEG,
    (NEG, POS): TOP,
    (NEG, TOP): TOP,
    (ZERO, NEG): NEG,
    (ZERO, ZERO): ZERO,
    (ZERO, POS): POS,
    (ZERO, TOP): TOP,
    (POS, NEG): TOP,
    (POS, ZERO): POS,
    (POS, POS): POS,
    (POS, TOP): TOP,
    (TOP, NEG): TOP,
    (TOP, ZERO): TOP,
    (TOP, POS): TOP,
    (TOP, TOP): TOP,
}


def _neg(a: str) -> str:
    return {NEG: POS, ZERO: ZERO, POS: NEG, TOP: TOP}[a]


def _abs_eval(e: object, env: dict[str, str]) -> str:
    if isinstance(e, tuple):
        op = e[0]
        if op == "lit":
            v = float(e[1])
            return POS if v > 0 else NEG if v < 0 else ZERO
        if op == "var":
            return env[e[1]]
        if op == "neg":
            return _neg(_abs_eval(e[1], env))
        a, b = _abs_eval(e[1], env), _abs_eval(e[2], env)
        if op == "mul":
            return _MUL[(a, b)]
        if op == "add":
            return _ADD[(a, b)]
        if op == "sub":
            return _ADD[(a, _neg(b))]
    raise ValueError(e)


def _concrete_sign(v: float) -> str:
    return POS if v > 0 else NEG if v < 0 else ZERO


def _concretize(sgn: str, rng: np.random.Generator) -> float:
    r = rng
    if sgn == POS:
        return float(r.uniform(0.01, 100.0))
    if sgn == NEG:
        return float(-r.uniform(0.01, 100.0))
    return 0.0


def _covers(abstract: str, concrete: float) -> bool:
    cs = _concrete_sign(concrete)
    return abstract in (TOP, cs) or (abstract == "0" and concrete == 0.0)


def _ceval(e: object, env: dict[str, float]) -> float:
    if isinstance(e, tuple):
        if e[0] == "lit":
            return float(e[1])
        if e[0] == "var":
            return env[e[1]]
        if e[0] == "neg":
            return -_ceval(e[1], env)
        a, b = _ceval(e[1], env), _ceval(e[2], env)
        return {"mul": a * b, "add": a + b, "sub": a - b}[e[0]]
    raise ValueError(e)


def bench_sign_domain(seed: int = _SEED) -> dict[str, float]:
    import numpy as np

    checks: list[bool] = []
    # multiplication table spot checks
    checks.append(_MUL[(NEG, POS)] == NEG and _MUL[(NEG, NEG)] == POS)
    checks.append(_ADD[(POS, NEG)] == TOP and _ADD[(ZERO, POS)] == POS)
    # abstract transfer on (-)*(-) -> +
    e1 = ("mul", ("var", "a"), ("var", "a"))
    checks.append(_abs_eval(e1, {"a": NEG}) == POS)
    # x + (-x) via TOP imprecision vs concrete 0 — abstract may be TOP (sound)
    e2 = ("add", ("var", "x"), ("neg", ("var", "x")))
    checks.append(_abs_eval(e2, {"x": TOP}) == TOP)
    # soundness: random concretizations always covered
    rng: np.random.Generator = np.random.default_rng(seed)
    e3 = ("mul", ("add", ("var", "p"), ("var", "q")), ("neg", ("var", "r")))
    env_a = {"p": POS, "q": ZERO, "r": POS}
    a = _abs_eval(e3, env_a)  # (pos+0)*(-pos) = -
    checks.append(a == NEG)
    for _ in range(60):
        env_c: dict[str, float] = {k: _concretize(v, rng) for k, v in env_a.items()}
        checks.append(_covers(a, _ceval(e3, env_c)))
    return {"synthetic_sign_domain": float(sum(checks)) / len(checks)}
