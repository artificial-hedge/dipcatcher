"""Representable presheaves on a 2-object category (SYNTHETIC)."""

from __future__ import annotations


class TwoObjCat:
    """Category with objects {A,B}: Hom(A,A)={id}, Hom(B,B)={id},
    Hom(A,B) = 2 maps {f,g}, Hom(B,A) = empty."""

    objs = ("A", "B")

    @staticmethod
    def hom(x: str, y: str) -> int:
        table = {("A", "A"): 1, ("B", "B"): 1, ("A", "B"): 2, ("B", "A"): 0}
        return table[(x, y)]


def yoneda_sizes(rep_obj: str) -> dict[str, int]:
    """Representable presheaf Hom(-, rep_obj): sizes at each object."""
    return {x: TwoObjCat.hom(x, rep_obj) for x in TwoObjCat.objs}


def _bench_presheaf(seed: int = 0) -> float:
    checks = []
    # Yoneda A: Hom(A,A)=1, Hom(B,A)=0
    ya = yoneda_sizes("A")
    checks.append(ya == {"A": 1, "B": 0})
    # Yoneda B: Hom(A,B)=2, Hom(B,B)=1
    yb = yoneda_sizes("B")
    checks.append(yb == {"A": 2, "B": 1})
    # presheaf is contravariant: pullback along f: A->B maps Hom(B,B)->Hom(A,B)
    checks.append(TwoObjCat.hom("A", "B") == TwoObjCat.hom("B", "B") * 2)
    # representables are distinct (Yoneda embedding faithful)
    checks.append(ya != yb)
    # Hom counts nonnegative
    checks.append(all(v >= 0 for v in ya.values()))
    return float(sum(checks) / len(checks))


def bench_presheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_presheaf": _bench_presheaf(seed)}
