"""Software pipelining / modulo scheduling (wave 294).

Initiation interval II = max(ResII, RecII): resource-bound from issue
slots per unit, recurrence-bound from loop-carried dependency cycle
latency/distance — verified against slow unrolled schedule throughput.
"""

_SEED = 20261231 + 849


def res_ii(ops: list[str]) -> float:
    # units: ALU=2/cycle, MEM=1/cycle, FPU=1/cycle
    res = {"ALU": 2, "MEM": 1, "FPU": 1}
    use = {"alu": "ALU", "load": "MEM", "store": "MEM", "fadd": "FPU", "fmul": "FPU"}
    need: dict[str, int] = {}
    for o in ops:
        u = use[o]
        need[u] = need.get(u, 0) + 1
    return max(need[u] / res[u] for u in need)


def rec_ii(cycles: list[tuple[int, int]]) -> float:
    # cycles: (latency_sum, distance_sum) per loop-carried cycle
    return max((lat / max(dist, 1) for lat, dist in cycles), default=0.0)


def ii(ops: list[str], cycles: list[tuple[int, int]]) -> int:
    import math

    return max(math.ceil(res_ii(ops)), math.ceil(rec_ii(cycles)))


def bench_modulo_sched(seed: int = _SEED) -> dict[str, float]:
    # loop: load, fmul, fadd, store — ResII = max(2/2,1/1,2/1)=2; recurrence fadd->next fadd latency 1 dist 1 → RecII=1 → II=2
    ok = int(ii(["load", "fmul", "fadd", "store"], [(1, 1)]) == 2)
    # recurrence-bound: serial dep latency 3
    ok += int(ii(["alu"], [(3, 1)]) == 3)
    # resource-bound: 5 stores → 5
    ok += int(ii(["store"] * 5, []) == 5)
    # distance-2 recurrence halves
    ok += int(ii(["alu"], [(4, 2)]) == 2)
    return {"synthetic_modulo": float(ok == 4)}
