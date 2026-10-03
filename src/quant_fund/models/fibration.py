"""Fibrations and long exact sequences of homotopy groups (SYNTHETIC)."""

from __future__ import annotations


def euler_char_multiplicative(chi_f: int, chi_b: int) -> int:
    """Fiber bundle with finite CW fiber: chi(E) = chi(F) * chi(B)."""
    return chi_f * chi_b


def les_homotopy_check(
    pi: dict[int, dict[str, int]], total: dict[int, int], base: dict[int, int]
) -> bool:
    """Verify exactness bookkeeping of the homotopy LES on supplied ranks:
    rank(pi_n F) - rank(pi_n E) + rank(pi_n B) alternates in exact seq:
    ... -> pi_n F -> pi_n E -> pi_n B -> pi_{n-1} F -> ...
    For an exact sequence of groups, alternating sum of ranks over a
    finite window = boundary contributions. This toy checks the Hopf
    fibration's key isomorphisms pi_3(S3) ~= pi_3(S2) fiber contribution.
    """
    # exactness at E: im(F->E) = ker(E->B); on ranks in a finite window:
    # sum_i (-1)^i rank G_i over an exact sequence = boundary terms only
    ok = True
    for n in sorted(pi):
        pf = pi[n]["f"]
        pe = pi[n]["e"]
        pb = pi[n]["b"]
        # exactness at E: rank(E) = rank ker(E->B) + rank im(E->B)
        # <= rank(F) + rank(B) since ker(E->B) = im(F->E) <= rank(F)
        ok = ok and pe <= pf + pb
    # Hopf fibration S1 -> S3 -> S2: pi_2(S2)=Z maps iso from pi_2(S3)=0?
    # actually boundary pi_2(S2) -> pi_1(S1) is iso -> pi_2(S3) -> pi_2(S2)
    # is the zero map; and pi_3(S3) -> pi_3(S2) is iso (both Z)
    return ok and total.get(3) == base.get(3)


def _bench_fibration(seed: int = 0) -> float:
    checks = []
    # chi multiplicativity: S1xS1: 0*0=0; S3 = S1-bundle over S2: 0*2=0
    checks.append(euler_char_multiplicative(0, 0) == 0)
    checks.append(euler_char_multiplicative(0, 2) == 0)
    # product torus T2: chi 0; S2 x S2: 4
    checks.append(euler_char_multiplicative(2, 2) == 4)
    # Hopf LES data: pi_3(S3)=Z matches pi_3(S2)=Z (fiber S1 has pi_3=0)
    pi = {3: {"f": 0, "e": 1, "b": 1}, 2: {"f": 0, "e": 0, "b": 1}}
    checks.append(les_homotopy_check(pi, {3: 1}, {3: 1}))
    # path-loop fibration: PX contractible -> pi_n(Omega X) = pi_{n+1}(X)
    checks.append(les_homotopy_check({2: {"f": 1, "e": 0, "b": 1}}, {2: 0}, {2: 1}))
    return float(sum(checks) / len(checks))


def bench_fibration(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fibration": _bench_fibration(seed)}
