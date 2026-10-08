"""Functor axiom checker (wave 289) (SYNTHETIC).

F: C -> D preserves identities F(id_A)=id_{FA} and composition
F(g f) = Fg Ff — verified exhaustively on small homomorphisms.
"""

_SEED = 20261231 + 819


def is_functor(
    src_c: dict[int, int],
    comp_c: dict[tuple[int, int], int],
    ident_c: dict[int, int],
    fobj: dict[int, int],
    farr: dict[int, int],
    comp_d: dict[tuple[int, int], int],
    ident_d: dict[int, int],
) -> bool:
    # identities
    for o, i in ident_c.items():
        if farr[i] != ident_d.get(fobj[o]):
            return False
    # composition
    for (f, g), fg in comp_c.items():
        lhs = farr[fg]
        rhs = comp_d.get((farr[f], farr[g]))
        if lhs != rhs:
            return False
    return True


def bench_functor_check(seed: int = _SEED) -> dict[str, float]:
    # C = Z2 monoid cat (same as fin_cat); D = Z4 monoid; F embeds g->g2
    ok = is_functor(
        {0: 0, 1: 0},
        {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        {0: 0},
        {0: 0},
        {0: 0, 1: 2},
        {(a, b): (a + b) % 4 for a in range(4) for b in range(4)},
        {0: 0},
    )
    # non-homomorphism g->g (identity on elements) fails: F(g*g)=F(e)=0 vs Fg Fg = g*g = e -> works actually for Z2->Z4 id? 1->1: F(g²)=F(1)=0? comp_d(1,1)=2; lhs=F(comp(1,1))=F(0)=0; rhs=2 → fail
    bad = is_functor(
        {0: 0, 1: 0},
        {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        {0: 0},
        {0: 0},
        {0: 0, 1: 1},
        {(a, b): (a + b) % 4 for a in range(4) for b in range(4)},
        {0: 0},
    )
    return {"synthetic_functor": float(ok and not bad)}
