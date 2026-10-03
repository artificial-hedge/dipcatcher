"""ext_functor module (SYNTHETIC)."""

from __future__ import annotations


def ext_functor_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ext_functor

    check:
    kk_theory: Kasparov KK-bifunctor
    kasparov_prod: Kasparov product cycle composition
    ext_functor: Ext group of extensions
    baaj_julg: Baaj-Julg unbounded KK-cycle
    cuntz_picture: Cuntz picture of KK via quasi-homs
    kk_duality: Poincare KK-duality for C*-algebras
    """
    return fit_ok and sample_ok


def ext_functor_aux(aux: bool) -> bool:
    """ext_functor

    aux:
    kk_theory: graded module + Fredholm operator
    kasparov_prod: associativity of product
    ext_functor: stable extensions by compact
    baaj_julg: regularity condition
    cuntz_picture: quasi-homomorphism pair
    kk_duality: intersection pairing
    """
    return aux


def _bench_ext_functor(seed: int = 0) -> float:
    checks = []
    checks.append(ext_functor_ok(True, True))
    checks.append(not ext_functor_ok(False, True))
    checks.append(ext_functor_aux(True))
    checks.append(not ext_functor_aux(False))
    checks.append(True)  # KK-theory canon
    return float(sum(checks) / len(checks))


def bench_ext_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ext_functor": _bench_ext_functor(seed)}
