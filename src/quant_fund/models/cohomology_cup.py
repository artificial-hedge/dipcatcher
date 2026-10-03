"""Cup product on simplicial cochains: graded-commutativity check (SYNTHETIC)."""

from __future__ import annotations

Simplex = tuple[int, ...]


def cochain_eval(cochain: dict[Simplex, int], s: Simplex) -> int:
    return cochain.get(tuple(sorted(s)), 0)


def coboundary(c: dict[Simplex, int], k: int, simplices: list[Simplex]) -> dict[Simplex, int]:
    """(dc)(v0..v_{k+1}) = sum_i (-1)^i c(face_i)."""
    out: dict[Simplex, int] = {}
    for s in simplices:
        if len(s) != k + 2:
            continue
        total = 0
        for i in range(len(s)):
            face = s[:i] + s[i + 1 :]
            total += ((-1) ** i) * cochain_eval(c, face)
        out[s] = total
    return out


def cup(
    c1: dict[Simplex, int], c2: dict[Simplex, int], k1: int, k2: int, simplices: list[Simplex]
) -> dict[Simplex, int]:
    """(c1~c2)(v0..v_{k1+k2}) = c1(v0..v_k1) * c2(v_k1..v_{k1+k2})."""
    out: dict[Simplex, int] = {}
    for s in simplices:
        if len(s) != k1 + k2 + 1:
            continue
        front = s[: k1 + 1]
        back = s[k1 : k1 + k2 + 1]
        out[s] = cochain_eval(c1, front) * cochain_eval(c2, back)
    return out


def _bench_cohomology_cup(seed: int = 0) -> float:
    checks = []
    # triangle: 0-simplices {0},{1},{2}, 1-simplices (01),(12),(02), 2-simplex (012)
    simps: list[Simplex] = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    c0: dict[Simplex, int] = {(0,): 1, (1,): 0, (2,): 0}
    d0 = coboundary(c0, 0, simps)
    checks.append(d0.get((0, 1)) == -1 and d0.get((0, 2)) == -1 and d0.get((1, 2)) == 0)
    # d^2 = 0
    d1 = coboundary(d0, 1, simps)
    checks.append(all(v == 0 for v in d1.values()))
    # cup: c0 ~ c0 at vertex 0 -> on (0,1): c0(0)*c0(1) = 0
    cc = cup(c0, c0, 0, 0, simps)
    checks.append(cc.get((0, 1), 0) == 0)
    c01: dict[Simplex, int] = {(0, 1): 1}
    cc2 = cup(c0, c01, 0, 1, simps)
    checks.append(cc2.get((0, 1), 0) == 1 and cc2.get((0, 2), 0) == 0)  # c0(0)*c01(0,1)=1
    checks.append(cup(c01, c0, 1, 0, simps).get((0, 1), 0) == 0)  # c01(0,1)*c0(1)=0
    return float(sum(checks) / len(checks))


def bench_cohomology_cup(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cohomology_cup": _bench_cohomology_cup(seed)}
