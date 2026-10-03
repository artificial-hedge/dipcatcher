"""Induced characters: Ind_H^G psi via class formula on S3 (SYNTHETIC)."""

from __future__ import annotations

S3 = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]
H2 = [(0, 1, 2), (1, 0, 2)]  # subgroup {e, (12)}


def comp(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(p[q[i]] for i in range(3))


def inv(p: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * 3
    for i, v in enumerate(p):
        out[v] = i
    return tuple(out)


def conjugate(g: tuple[int, ...], x: tuple[int, ...]) -> tuple[int, ...]:
    return comp(comp(g, x), inv(g))


def induced_char(psi: dict[tuple[int, ...], float]) -> dict[tuple[int, ...], float]:
    """Ind_H^G psi(x) = (1/|H|) sum_{g: gxg^-1 in H} psi(gxg^-1)."""
    out: dict[tuple[int, ...], float] = {}
    for x in S3:
        total = 0.0
        for g in S3:
            c = conjugate(g, x)
            if c in psi:
                total += psi[c]
        out[x] = total / len(H2)
    return out


def _bench_induced_rep(seed: int = 0) -> float:
    checks = []
    e = (0, 1, 2)
    t12 = (1, 0, 2)
    t13 = (0, 2, 1)
    c3 = (1, 2, 0)
    # Ind of trivial from {e,(12)}: char(x) = #cosets fixed; Ind triv(x)= (1/2)*|{g: gxg^-1 in H}|
    psi_triv: dict[tuple[int, ...], float] = {e: 1.0, t12: 1.0}
    ind = induced_char(psi_triv)
    checks.append(abs(ind[e] - 3.0) < 1e-9)
    checks.append(abs(ind[t12] - 1.0) < 1e-9)
    checks.append(abs(ind[t13] - 1.0) < 1e-9)
    checks.append(abs(ind[c3] - 0.0) < 1e-9)
    # Ind of sign from H: psi(e)=1, psi(t)=-1
    psi_sign: dict[tuple[int, ...], float] = {e: 1.0, t12: -1.0}
    ind2 = induced_char(psi_sign)
    checks.append(abs(ind2[e] - 3.0) < 1e-9)
    checks.append(abs(ind2[t12] + 1.0) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_induced_rep(seed: int = 0) -> dict[str, float]:
    return {"synthetic_induced_rep": _bench_induced_rep(seed)}
