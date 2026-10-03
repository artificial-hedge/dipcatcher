"""Galois group of finite-field extensions: Frobenius cyclic structure (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.field_ext import padd, pdivmod, peval, pmod, pmul

Poly = list[int]


def ext_pow(x: Poly, k: int, p: int, mod: Poly) -> Poly:
    acc: Poly = [1]
    base = x
    while k:
        if k & 1:
            acc = pdivmod(pmul(acc, base, p), mod, p)[1]
        base = pdivmod(pmul(base, base, p), mod, p)[1]
        k >>= 1
    return acc


def frobenius(x: Poly, p: int, mod: Poly) -> Poly:
    """x -> x^p in GF(p)[x]/(mod): the generating automorphism."""
    return ext_pow(x, p, p, mod)


def automorphisms(p: int, mod: Poly) -> list[Poly]:
    """Images of alpha=x under all automorphisms (powers of Frobenius)."""
    alpha: Poly = [0, 1]
    deg = len(mod) - 1
    return [ext_pow(alpha, p**i, p, mod) for i in range(deg)]


def is_automorphism_root(img: Poly, p: int, mod: Poly) -> bool:
    return peval(mod, img, p, mod) == [0]


def _bench_galois_group(seed: int = 0) -> float:
    checks = []
    p, mod = 2, [1, 1, 1]  # GF(4)
    alpha = [0, 1]
    f1 = frobenius(alpha, p, mod)
    checks.append(f1 == pmod(padd(alpha, [1], p), p))  # sigma(alpha)=alpha^2=alpha+1
    f2 = frobenius(f1, p, mod)
    checks.append(f2 == alpha)  # sigma^2 = id: Gal(GF4/GF2) = Z/2
    autos = automorphisms(p, mod)
    checks.append(len(autos) == 2)
    checks.append(all(is_automorphism_root(a_, p, mod) for a_ in autos))
    # fixed field of Frobenius = GF(2): sigma(x)=x on basis {0,1}
    checks.append(frobenius([1], p, mod) == [1])
    checks.append(frobenius([0], p, mod) == [0])
    return float(sum(checks) / len(checks))


def bench_galois_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galois_group": _bench_galois_group(seed)}
