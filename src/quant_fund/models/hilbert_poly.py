"""Hilbert function of monomial ideals: count of surviving monomials by degree (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.monomial_ideal import in_ideal


def monomials_deg(d: int, nvars: int) -> list[tuple[int, ...]]:
    """Exponent vectors of total degree d in nvars."""
    if nvars == 1:
        return [(d,)]
    out = []
    for i in range(d + 1):
        for rest in monomials_deg(d - i, nvars - 1):
            out.append((i,) + rest)
    return out


def hilbert_fn(ideal_gens: frozenset[tuple[int, ...]], nvars: int, max_deg: int) -> list[int]:
    """H(M)(d) = # monomials of deg d not in the ideal (quotient ring)."""
    return [
        sum(1 for m in monomials_deg(d, nvars) if not in_ideal(m, ideal_gens))
        for d in range(max_deg + 1)
    ]


def _bench_hilbert_poly(seed: int = 0) -> float:
    checks = []
    # k[x,y]/<x^2>: H = 1,2,2,2,... (deg-d monomials: x^0 y^d, x^1 y^{d-1} survive for d>=1)
    hf = hilbert_fn(frozenset({(2, 0)}), 2, 4)
    checks.append(hf == [1, 2, 2, 2, 2])
    # k[x,y]/<x,y> = k: H = 1,0,0,...
    checks.append(hilbert_fn(frozenset({(1, 0), (0, 1)}), 2, 3) == [1, 0, 0, 0])
    # k[x,y]/<x^2,y^2>: H = 1,2,1,0
    checks.append(hilbert_fn(frozenset({(2, 0), (0, 2)}), 2, 3) == [1, 2, 1, 0])
    # no ideal: H(d) = d+1 for 2 vars
    checks.append(hilbert_fn(frozenset(), 2, 3) == [1, 2, 3, 4])
    # monomials count deg 2 in 2 vars = 3
    checks.append(len(monomials_deg(2, 2)) == 3)
    # k[x]/<x^3>: H = 1,1,1,0,0
    checks.append(hilbert_fn(frozenset({(3,)}), 1, 4) == [1, 1, 1, 0, 0])
    return float(sum(checks) / len(checks))


def bench_hilbert_poly(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hilbert_poly": _bench_hilbert_poly(seed)}
