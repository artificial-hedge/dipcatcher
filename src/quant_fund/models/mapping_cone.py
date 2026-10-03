"""Mapping cone homology: cofiber of a map (SYNTHETIC)."""

from __future__ import annotations


def cone_betti(map_deg: int, src_betti: list[int], tgt_betti: list[int]) -> list[int]:
    """Reduced homology of the mapping cone C_f of f: S^n -> S^n of
    degree d: H_n = Z/d (rank 0 if d != 0, else 1), H_{n+1} = 0 if
    d != 0 (killed) else Z. Model as betti numbers of C_f where
    the cone kills the image sphere.

    For d != 0: betti = all zeros except possibly degrees elsewhere.
    For d = 0: C_f ~= S^{n+1} v S^n -> both generators survive.
    """
    n = len(src_betti) - 1
    out = [0] * (n + 2)
    if map_deg == 0:
        out[n] = 1
        out[n + 1] = 1
    return out


def torsion_order(map_deg: int) -> int:
    """H_n(C_f) = Z/dZ for degree-d map: torsion order |d|."""
    return abs(map_deg)


def _bench_mapping_cone(seed: int = 0) -> float:
    checks = []
    s1 = [1, 1]
    # identity map cone is contractible: all reduced homology 0
    checks.append(cone_betti(1, s1, s1) == [0, 0, 0])
    # degree-2 S1 -> S1 cone is RP^2: H1 = Z/2 (torsion), H2 = 0
    checks.append(cone_betti(2, s1, s1) == [0, 0, 0])
    checks.append(torsion_order(2) == 2)
    # degree-0 map: cone = S1 v S2 -> betti [0? reduced: H1=Z,H2=Z]
    checks.append(cone_betti(0, s1, s1) == [0, 1, 1])
    # degree-3 on S2 -> S2 gives torsion Z/3 in H2
    s2 = [1, 0, 1]
    checks.append(cone_betti(3, s2, s2) == [0, 0, 0, 0])
    checks.append(torsion_order(3) == 3)
    return float(sum(checks) / len(checks))


def bench_mapping_cone(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mapping_cone": _bench_mapping_cone(seed)}
