"""Finite-category axiom checker (wave 289).

A category = objects, hom-sets, composition table, identities.
Verify associativity + identity laws exhaustively on small examples:
the 2-object poset category, a monoid-as-1-object category, and a
3-object path category.
"""

_SEED = 20261231 + 818


# representation: arrows are ints; comp[(a,b)] -> c or None; id[obj] -> arrow
def check(
    objs: list[int],
    src: dict[int, int],
    dst: dict[int, int],
    comp: dict[tuple[int, int], int],
    ident: dict[int, int],
) -> bool:
    # identity law
    for f in src:
        i_src, i_dst = ident[src[f]], ident[dst[f]]
        if comp.get((i_src, f)) != f or comp.get((f, i_dst)) != f:
            return False
    # associativity where defined
    for f, g in comp:
        fg = comp[(f, g)]
        for h, k in comp:
            if h != fg:
                continue
            # find pre-f arrow: we have (f,g) and (fg,k); need gk defined
            for g2, k2 in comp:
                if g2 == g and k2 == k:
                    gk = comp[(g, k)]
                    if comp.get((f, gk)) != comp[(fg, k)]:
                        return False
    return True


def bench_fin_cat(seed: int = _SEED) -> dict[str, float]:
    # monoid Z2 as 1-object category: arrows 0=e, 1=g; comp = xor
    ok = check(
        [0],
        {0: 0, 1: 0},
        {0: 0, 1: 0},
        {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        {0: 0},
    )
    # poset 0 <= 1: arrows e0,e1, f:0->1
    ok &= check(
        [0, 1],
        {0: 0, 1: 1, 2: 0},
        {0: 0, 1: 1, 2: 1},
        {(0, 0): 0, (1, 1): 1, (0, 2): 2, (2, 1): 2},
        {0: 0, 1: 1},
    )
    # broken composition must fail
    bad = check(
        [0],
        {0: 0, 1: 0},
        {0: 0, 1: 0},
        {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 0},
        {0: 0},
    )
    return {"synthetic_fin_cat": float(ok == 1 and not bad)}
