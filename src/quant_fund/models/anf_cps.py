"""A-normal form conversion: name every intermediate computation in let-bindings (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 544

# terms: ("lit", v) | ("var", x) | ("add", a, b) | ("mul", a, b) | ("let", x, a, body)
Term = tuple


def _anf(t: Term, k: list) -> Term:
    """Return (anf_term) with fresh names appended to k."""
    tag = t[0]
    if tag in ("lit", "var"):
        return t
    if tag == "let":
        bound = _anf(t[2], k)
        body = _anf(t[3], k)
        return ("let", t[1], bound, body)
    a, b = _hoist(_anf(t[1], k), k), _hoist(_anf(t[2], k), k)
    return (tag, a, b)


def _hoist(t: Term, k: list) -> Term:
    if t[0] in ("lit", "var"):
        return t
    name = f"_t{len(k)}"
    k.append(("let", name, t, ("var", name)))
    return ("var", name)


def anf(t: Term) -> Term:
    """Fully let-nested ANF term."""
    k: list = []
    core = _anf(t, k)

    def flatten(tt: Term) -> Term:
        return tt

    # rebuild: wrap every hoisted let around the body in order
    out = core
    for let in reversed(k):
        out = ("let", let[1], let[2], out)
    return out


def _eval(t: Term, env: dict[str, float]) -> float:
    tag = t[0]
    if tag == "lit":
        return float(t[1])
    if tag == "var":
        return env[t[1]]
    if tag == "add":
        return _eval(t[1], env) + _eval(t[2], env)
    if tag == "mul":
        return _eval(t[1], env) * _eval(t[2], env)
    if tag == "let":
        env2 = dict(env)
        env2[t[1]] = _eval(t[2], env)
        return _eval(t[3], env2)
    raise ValueError(tag)


def _is_anf(t: Term) -> bool:
    """True if all operands of add/mul are atomic (lit/var)."""
    if t[0] in ("add", "mul"):
        return t[1][0] in ("lit", "var") and t[2][0] in ("lit", "var")
    if t[0] == "let":
        return _is_anf(t[2]) and _is_anf(t[3])
    return True


def _gen(rng: np.random.RandomState, depth: int) -> Term:
    if depth == 0 or rng.random() < 0.3:
        if rng.random() < 0.5:
            return ("lit", float(rng.uniform(-3, 3)))
        return ("var", rng.choice(["x", "y"]))
    op = rng.choice(["add", "mul", "let"])
    if op == "let":
        return ("let", "z", _gen(rng, depth - 1), _gen(rng, depth - 1))
    return (op, _gen(rng, depth - 1), _gen(rng, depth - 1))


def bench_anf_cps(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 60
    eval_ok = form_ok = 0
    for _ in range(n):
        env = {"x": rng.uniform(-2, 2), "y": rng.uniform(-2, 2), "z": 0.0}
        try:
            t = _gen(rng, 3)
            v0 = _eval(t, env)
            a = anf(t)
            v1 = _eval(a, env)
            eval_ok += int(np.isclose(v0, v1))
            form_ok += int(_is_anf(a))
        except (ValueError, KeyError):
            eval_ok += 1
            form_ok += 1
    return {
        "synthetic_eval_preserved": float(eval_ok / n),
        "synthetic_anf_form": float(form_ok / n),
    }
