"""Constant folding + propagation over straight-line SSA."""

import numpy as np

_SEED = 20261231 + 725

# stmts: (var, "const", v) | (var, op, a, b) with a,b var-or-const refs
Stmt = tuple


def fold(stmts: list[Stmt]) -> dict[str, int]:
    env: dict[str, int] = {}

    def val(t: object) -> int | None:
        if isinstance(t, int):
            return t
        if isinstance(t, str) and t in env:
            return env[t]
        return None

    for st in stmts:
        name = st[0]
        if st[1] == "const":
            env[name] = int(st[2])
            continue
        a, b = val(st[2]), val(st[3])
        if a is not None and b is not None:
            env[name] = a + b if st[1] == "add" else a * b
    return env


def bench_const_fold(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        n = int(rng.randint(4, 10))
        stmts: list[Stmt] = []
        # build chain where all inputs resolve
        stmts.append(("v0", "const", int(rng.randint(-5, 5))))
        for i in range(1, n):
            op = "add" if rng.rand() < 0.5 else "mul"
            a = f"v{rng.randint(0, i)}" if rng.rand() < 0.7 else int(rng.randint(-3, 3))
            b = f"v{rng.randint(0, i)}" if rng.rand() < 0.7 else int(rng.randint(-3, 3))
            stmts.append((f"v{i}", op, a, b))
        env = fold(stmts)
        # oracle: evaluate sequentially with dict
        ev: dict[str, int] = {}
        expect = True
        for st in stmts:
            if st[1] == "const":
                ev[st[0]] = int(st[2])
            else:
                va = ev[st[2]] if isinstance(st[2], str) else int(st[2])
                vb = ev[st[3]] if isinstance(st[3], str) else int(st[3])
                ev[st[0]] = va + vb if st[1] == "add" else va * vb
            if env.get(st[0]) != ev[st[0]]:
                expect = False
        ok += float(expect)
    return {"synthetic_fold_exact": ok / trials}
