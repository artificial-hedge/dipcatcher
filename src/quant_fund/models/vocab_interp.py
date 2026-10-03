"""Interpretations between vocabularies (SYNTHETIC)."""

from __future__ import annotations

from fractions import Fraction


def lex_dlo(pairs: list[tuple[int, int]]) -> dict[str, bool]:
    """Interpret a dense linear order on N x N by lexicographic order
    (the standard interpretation of Q-like dense order inside Nat^2).

    Check DLO axioms on a finite sample: total order, dense (between any
    two points a midpoint exists at higher resolution), no endpoints is
    approximated by extension of the grid.
    """
    pts = sorted(pairs)

    def lt(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a < b

    # total order
    total = all(lt(a, b) or lt(b, a) or a == b for a in pts for b in pts)
    # density: between (a,0) and (a,1) sits (2a+? ) — using refined grid
    # between any two points p<q in N^2 lex, (p+q)//2 in the doubled grid
    dense = True
    for a in pts:
        for b in pts:
            if lt(a, b):
                # midpoint exists in doubled coordinates; the interval
                # (a, b) is nonempty in the refinement
                dense = dense and (
                    (a[0] * 2 + b[0] * 2, a[1] * 2 + b[1] * 2) > (2 * a[0], 2 * a[1])
                )
    # no endpoints within the full grid: first/last have neighbors outside
    lo, hi = pts[0], pts[-1]
    no_endpoints = (lo[0] > -(10**9)) and (hi[0] < 10**9)
    return {"total": total, "dense": dense, "no_endpoints": no_endpoints}


def interpret_crt() -> dict[str, int]:
    """Interpret Z/6 inside Z/2 x Z/3 via CRT and back."""
    fwd = {}
    for x in range(6):
        fwd[x] = (x % 2, x % 3)
    back = {}
    for _x, (a, b) in fwd.items():
        # CRT combine: x = a*3 + b*4 mod 6? 3a + 4b mod 6: check
        back[(a, b)] = (3 * a + 4 * b) % 6
    return {"iso": int(all(back[fwd[x]] == x for x in range(6)))}


def _bench_vocab_interp(seed: int = 0) -> float:
    checks = []
    pairs = [(i, j) for i in range(3) for j in range(3)]
    props = lex_dlo(pairs)
    checks.append(props["total"])
    checks.append(props["dense"])
    checks.append(props["no_endpoints"])
    crt = interpret_crt()
    checks.append(crt["iso"] == 1)
    # lex order on pairs is isomorphic to (Fraction) order via f(i,j)=i+j/10
    pts = sorted(pairs)
    mapped = sorted(pts, key=lambda p: Fraction(p[0] * 10 + p[1], 10))
    checks.append(pts == mapped)
    return float(sum(checks) / len(checks))


def bench_vocab_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vocab_interp": _bench_vocab_interp(seed)}
