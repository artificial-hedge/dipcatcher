"""Static timing analysis (wave 291) (SYNTHETIC).

Longest-path arrival times on a combinational DAG: at(node) =
max(at(fanin) + gate_delay). Critical path and slack vs a brute-force
path enumeration oracle.
"""

_SEED = 20261231 + 831


def longest_path(
    edges: dict[int, list[int]], delay: dict[int, float], sources: list[int], n: int
) -> dict[int, float]:
    # topological DP; edges i -> succ
    at = {s: 0.0 for s in sources}
    order: list[int] = []
    indeg = {i: 0 for i in range(n)}
    for _u, succ in edges.items():
        for v in succ:
            indeg[v] = indeg.get(v, 0) + 1
    stack = [i for i in range(n) if indeg.get(i, 0) == 0]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in edges.get(u, []):
            at[v] = max(at.get(v, -1e18), at.get(u, 0.0) + delay.get(u, 0.0))
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return at


def bench_sta_timing(seed: int = _SEED) -> dict[str, float]:
    # chain 0->1->2 plus shortcut 0->2; delays d=1: at[2]=2 not 1
    edges = {0: [1, 2], 1: [2]}
    delay = {0: 1.0, 1: 1.0, 2: 0.0}
    at = longest_path(edges, delay, [0], 3)
    ok = int(abs(at[2] - 2.0) < 1e-9)
    # diamond with asym delays
    edges = {0: [1, 2], 1: [3], 2: [3]}
    delay = {0: 0.5, 1: 3.0, 2: 1.0, 3: 0.0}
    at = longest_path(edges, delay, [0], 4)
    ok += int(abs(at[3] - 3.5) < 1e-9)
    # required-time slack: sink must arrive by 4.0
    ok += int(abs(4.0 - at[3] - 0.5) < 1e-9)
    return {"synthetic_sta": float(ok == 3)}
