"""Multivariate polynomial division mod a generating set (wave 281).

One-variable-reduction division for monomial ideals in k[x,y]: a monomial
x^a y^b is in <x^i y^j> iff i<=a and j<=b. Division reduces f term-by-term;
remainder zero <=> f in the monomial ideal (exact oracle).
"""

_SEED = 20261231 + 774

Term = tuple[int, int, int]  # (coef, a, b) for x^a y^b


def reduce_poly(f: list[Term], gens: list[Term]) -> list[Term]:
    rem = []
    for c, a, b in f:
        if any(gb <= b and ga <= a for _, ga, gb in gens):
            continue
        rem.append((c, a, b))
    return rem


def in_ideal(f: list[Term], gens: list[Term]) -> bool:
    return len(reduce_poly(f, gens)) == 0


def bench_ideal_member(seed: int = _SEED) -> dict[str, float]:
    gens = [(1, 2, 0), (1, 1, 1), (1, 0, 3)]  # <x^2, xy, y^3>
    # f = x^3 + x^2y + y^3 -> all terms divisible
    f_in = [(1, 3, 0), (1, 2, 1), (1, 0, 3)]
    # g = x + y^2 + x^2y -> x and y^2 survive
    g_out = [(1, 1, 0), (1, 0, 2), (1, 1, 1)]
    ok = int(in_ideal(f_in, gens))
    ok += int(not in_ideal(g_out, gens))
    rem = reduce_poly(g_out, gens)
    ok += int(sorted(rem) == sorted([(1, 1, 0), (1, 0, 2)]))
    return {"synthetic_ideal_div": float(ok == 3)}
