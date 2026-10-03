"""Reduced ordered BDD: build from a boolean formula on 4 vars via
Shannon expansion with a shared unique table, then count satisfying
assignments and compare vs truth table.
"""

import itertools

_FORMS: dict[str, object] = {
    "(a and b) or (c and d)": lambda a, b, c, d: (a and b) or (c and d),
    "(a or b) and (not c or d)": lambda a, b, c, d: (a or b) and (not c or d),
    "(a == b) and (c == d)": lambda a, b, c, d: (a == b) and (c == d),
}


def _eval(form: str, env: dict[str, bool]) -> bool:
    fn = _FORMS[form]
    return bool(fn(*[env[k] for k in "abcd"]))  # type: ignore[operator]


class _BDD:
    def __init__(self) -> None:
        self.nodes: dict[tuple[str, int, int], int] = {}
        self.var: dict[int, str] = {}
        self.lo: dict[int, int] = {}
        self.hi: dict[int, int] = {}
        self.cache: dict[tuple[str, tuple], int] = {}

    def mk(self, var: str, lo: int, hi: int) -> int:
        if lo == hi:
            return lo
        key = (var, lo, hi)
        if key in self.nodes:
            return self.nodes[key]
        idx = len(self.nodes) + 2  # 0=False, 1=True terminals
        self.nodes[key] = idx
        self.var[idx] = var
        self.lo[idx] = lo
        self.hi[idx] = hi
        return idx

    def build(self, form: str, order: list[str], env: dict[str, bool]) -> int:
        key = (form + str(order), tuple(sorted(env.items())))
        if key in self.cache:
            return self.cache[key]
        if not order:
            val = _eval(form, env)
            r = 1 if val else 0
        else:
            v = order[0]
            lo = self.build(form, order[1:], {**env, v: False})
            hi = self.build(form, order[1:], {**env, v: True})
            r = self.mk(v, lo, hi)
        self.cache[key] = r
        return r

    def sat_count(self, node: int) -> int:
        # count assignments over the full 4-var order via paths
        memo: dict[int, int] = {}

        def count(n: int) -> int:
            if n in memo:
                return memo[n]
            memo[n] = count(self.lo[n]) + count(self.hi[n])
            return memo[n]

        return int(count(node))


def bench_bdd_ops(seed: int = 5909) -> dict[str, float]:
    _ = seed
    forms = ["(a and b) or (c and d)", "(a or b) and (not c or d)", "(a == b) and (c == d)"]
    order = ["a", "b", "c", "d"]
    bdd = _BDD()
    agree = 0
    node_tot = 0
    for f in forms:
        root = bdd.build(f, order, {})

        # expand count with skipped vars: recursive over ordered vars
        def cnt(n: int, idx: int) -> int:
            if n in (0, 1):
                return int(n * (2 ** (4 - idx)))
            var = bdd.var[n]
            v_idx = order.index(var)
            skip = v_idx - idx
            return int((cnt(bdd.lo[n], v_idx + 1) + cnt(bdd.hi[n], v_idx + 1)) * (2**skip))

        est = cnt(root, 0)
        truth = sum(
            1
            for vals in itertools.product([False, True], repeat=4)
            if _eval(f, dict(zip("abcd", vals, strict=True)))
        )
        agree += int(est == truth)
        node_tot += len(bdd.nodes)
    return {
        "synthetic_bdd_agree": float(agree),
        "synthetic_bdd_nodes": float(node_tot),
        "synthetic_bdd_forms": float(len(forms)),
    }
