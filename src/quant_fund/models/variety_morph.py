"""Affine varieties over GF(p): point sets, morphisms, coordinate rings (SYNTHETIC)."""

from __future__ import annotations

import itertools


def variety(fs: list[list[tuple[tuple[int, ...], int]]], p: int, n_vars: int) -> set[tuple[int, ...]]:
    """Common zeros of polys (little-endian coeffs) in GF(p)^n."""
    out = set()
    for pt in itertools.product(range(p), repeat=n_vars):
        ok = True
        for f in fs:
            val = 0
            for mono, coef in f:
                m = 1
                for v_, e_ in zip(pt, mono, strict=True):
                    m = m * pow(v_, e_, p) % p
                val = (val + coef * m) % p
            if val != 0:
                ok = False
                break
        if ok:
            out.add(pt)
    return out


def morphi(
    us: list[list[tuple[tuple[int, ...], int]]], pt: tuple[int, ...], p: int
) -> tuple[int, ...]:
    """Polynomial map given by component polys (mono->coef)."""
    out = []
    for u in us:
        val = 0
        for mono, coef in u:
            m = 1
            for v_, e_ in zip(pt, mono, strict=True):
                m = m * pow(v_, e_, p) % p
            val = (val + coef * m) % p
        out.append(val)
    return tuple(out)


def is_morphism_image(v_src: set[tuple[int, ...]], v_tgt: set[tuple[int, ...]], us, p: int) -> bool:
    return all(morphi(us, pt, p) in v_tgt for pt in v_src)


def _bench_variety_morph(seed: int = 0) -> float:
    checks = []
    p = 5
    # parabola y = x^2 in A2: poly y - x^2 = 0 -> mono dict form
    f: list[list[tuple[tuple[int, ...], int]]] = [[((0, 1), 1), ((2, 0), p - 1)]]  # y - x^2
    v = variety(f, p, 2)
    checks.append((2, 4) in v and (2, 3) not in v)
    checks.append(len(v) == 5)  # one pt per x
    # Frobenius map x -> x^p fixes F_p points: variety mapped to itself
    fr = [[((5, 0), 1)], [((0, 5), 1)]]  # type: list[list[tuple[tuple[int, ...], int]]]
    checks.append(is_morphism_image(v, v, fr, p))
    # projection to x-axis hits all of A1
    proj: list[list[tuple[tuple[int, ...], int]]] = [[((1, 0), 1)]]
    checks.append(len({morphi(proj, pt, p)[0] for pt in v}) == 5)
    # union of axes: xy = 0
    checks.append(len(variety([[((1, 1), 1)]], p, 2)) == 2 * p - 1)
    return float(sum(checks) / len(checks))


def bench_variety_morph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_variety_morph": _bench_variety_morph(seed)}
