"""List-monad laws (wave 289).

Monad (T, eta, mu): T = List, eta x = [x], mu = flatten.
Left unit mu . T eta = id; right unit mu . eta_T = id;
associativity mu . T mu = mu . mu — checked on finite test lists.
"""

_SEED = 20261231 + 823


def _t_map(f, lst: list) -> list:
    return [f(x) for x in lst]


def _flatten(ll: list[list]) -> list:
    return [x for inner in ll for x in inner]


def _eta(x):
    return [x]


def _bind(lst: list, f) -> list:
    return _flatten(_t_map(f, lst))


def bench_monad_laws(seed: int = _SEED) -> dict[str, float]:
    ok = 0
    xs = [0, 1, 2, 3]

    def f(x: int) -> list[int]:
        return [x, (x + 1) % 4]

    def g(x: int) -> list[int]:
        return [x * x % 4]

    # left unit: bind(eta(a), f) = f(a)
    ok += int(all(_bind(_eta(a), f) == f(a) for a in xs))
    # right unit: bind(m, eta) = m
    m = [1, 2, 3]
    ok += int(_bind(m, _eta) == m)
    # associativity: bind(bind(m,f),g) = bind(m, lambda x: bind(f(x), g))
    lhs = _bind(_bind(m, f), g)
    rhs = _bind(m, lambda x: _bind(f(x), g))
    ok += int(lhs == rhs)
    return {"synthetic_monad": float(ok == 3)}
