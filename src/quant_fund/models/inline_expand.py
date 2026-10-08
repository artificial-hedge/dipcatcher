"""Function inlining: expand calls with argument substitution (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 727

# func: (param_name, body_expr) ; expr as in partial_eval
Func = tuple[str, tuple]


def subst(expr: tuple, name: str, rep: tuple) -> tuple:
    tag = expr[0]
    if tag in ("const",):
        return expr
    if tag == "var":
        return rep if expr[1] == name else expr
    return (tag, subst(expr[1], name, rep), subst(expr[2], name, rep))


def run_expr(expr: tuple, env: dict[str, int]) -> int:
    tag = expr[0]
    if tag == "const":
        return int(expr[1])
    if tag == "var":
        return env[expr[1]]
    lft = run_expr(expr[1], env)
    r = run_expr(expr[2], env)
    return lft + r if tag == "add" else lft * r


def inline_call(func: Func, arg: tuple) -> tuple:
    return subst(func[1], func[0], arg)


def bench_inline_expand(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    f: Func = ("a", ("mul", ("var", "a"), ("add", ("var", "a"), ("const", 1))))
    ok = 0.0
    trials = 40
    for _ in range(trials):
        arg: tuple = ("add", ("var", "x"), ("const", int(rng.randint(-4, 4))))
        body = inline_call(f, arg)
        x = int(rng.randint(-5, 5))
        expect = run_expr(arg, {"x": x}) * (run_expr(arg, {"x": x}) + 1)
        ok += float(run_expr(body, {"x": x}) == expect)
    return {"synthetic_inline_exact": ok / trials}
