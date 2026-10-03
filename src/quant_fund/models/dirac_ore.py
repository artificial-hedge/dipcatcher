"""Dirac/Ore sufficient conditions for Hamiltonian cycles (SYNTHETIC)."""

from __future__ import annotations


def min_degree(adj: list[set[int]]) -> int:
    return min(len(a) for a in adj)


def dirac_holds(adj: list[set[int]]) -> bool:
    n = len(adj)
    return n >= 3 and min_degree(adj) >= n / 2


def ore_holds(adj: list[set[int]]) -> bool:
    n = len(adj)
    return all(
        len(adj[u]) + len(adj[v]) >= n for u in range(n) for v in range(u + 1, n) if v not in adj[u]
    )


def has_hamilton_cycle(adj: list[set[int]]) -> bool:
    n = len(adj)
    path = [0]
    seen = {0}

    def rec() -> bool:
        if len(path) == n:
            return path[0] in adj[path[-1]]
        u = path[-1]
        for w in adj[u]:
            if w not in seen:
                path.append(w)
                seen.add(w)
                if rec():
                    return True
                path.pop()
                seen.discard(w)
        return False

    return rec()


def _mk(edges: list[tuple[int, int]], n: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _bench_dirac_ore(seed: int = 0) -> float:
    checks = []
    # circulant C8(1,2): degree 4 = n/2 -> Dirac holds, cycle exists
    c82 = _mk([(i, (i + j) % 8) for i in range(8) for j in (1, 2)], 8)
    checks.append(dirac_holds(c82) and has_hamilton_cycle(c82))
    # K_{4,4} has min deg 4 = n/2 -> Dirac holds
    k44 = _mk([(i, j + 4) for i in range(4) for j in range(4)], 8)
    checks.append(dirac_holds(k44) and has_hamilton_cycle(k44))
    # K6 minus a star at vertex 0 (edges 0-3,0-4,0-5 removed): deg(0)=2
    # so Dirac fails, but every nonadjacent pair sums to 6 -> Ore holds
    k6_star = _mk(
        [
            (a, b)
            for a in range(6)
            for b in range(a + 1, 6)
            if (a, b) not in {(0, 3), (0, 4), (0, 5)}
        ],
        6,
    )
    checks.append(not dirac_holds(k6_star))
    checks.append(ore_holds(k6_star))
    checks.append(has_hamilton_cycle(k6_star))
    # path graph: neither holds, no cycle
    path = _mk([(i, i + 1) for i in range(5)], 6)
    checks.append(not dirac_holds(path) and not ore_holds(path))
    checks.append(not has_hamilton_cycle(path))
    return float(sum(checks) / len(checks))


def bench_dirac_ore(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dirac_ore": _bench_dirac_ore(seed)}
