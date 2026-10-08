"""Graph cyclomatic number equals Betti-1 (wave 280) (SYNTHETIC).

For a 1-dimensional complex (a graph): b_1 = E - V + C where C is the number
of connected components — verified against the rank computation on random
graphs and forests.
"""

import numpy as np

from quant_fund.models.simp_betti import betti

_SEED = 20261231 + 768


def _components(n: int, edges: list[tuple[int, ...]]) -> int:
    par = list(range(n))

    def find(x: int) -> int:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for i, j in edges:
        par[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def bench_graph_h1(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    trials = 8
    for _ in range(trials):
        n = int(rng.randint(4, 9))
        edges: list[tuple[int, ...]] = [
            (i, j) for i in range(n) for j in range(i + 1, n) if rng.rand() < 0.4
        ]
        c = _components(n, edges)
        want = len(edges) - n + c
        comp: list[list[tuple[int, ...]]] = [[(i,) for i in range(n)], edges]
        got = betti(comp)[1]
        ok += int(got == want)
    return {"synthetic_graph_h1": float(ok == trials)}
