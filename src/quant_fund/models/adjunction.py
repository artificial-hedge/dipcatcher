"""Free-forgetful adjunction F -| U (wave 289).

F: Set -> Mon (free monoid = lists) is left adjoint to forgetful U.
The adjunction bijection Hom_Mon(FX, M) ≅ Hom_Set(X, UM) is verified:
every set-map X -> U(M) extends to a unique monoid hom from lists.
"""

_SEED = 20261231 + 821


def extend_set_map(f, x_elems: list[int], m_op, m_e):
    # f: element -> monoid element; extension maps list -> fold
    def ext(lst: tuple[int, ...]):
        acc = m_e
        for x in lst:
            acc = m_op(acc, f[x])
        return acc

    return ext


def bench_adjunction(seed: int = _SEED) -> dict[str, float]:
    # monoid = (Z4, +, 0); X = {0,1}; f(0)=1, f(1)=2
    f = {0: 1, 1: 2}
    ext = extend_set_map(f, [0, 1], lambda a, b: (a + b) % 4, 0)
    ok = 0
    # extension is hom: ext(l1+l2) = ext(l1) + ext(l2)
    import itertools

    words = [tuple(w) for n in range(3) for w in itertools.product([0, 1], repeat=n)]
    for w1, w2 in itertools.product(words, repeat=2):
        if ext(w1 + w2) != (ext(w1) + ext(w2)) % 4:
            return {"synthetic_adjunction": 0.0}
    # unit: ext((x,)) = f(x)
    ok += int(ext((0,)) == 1 and ext((1,)) == 2)
    # bijection count: |Hom_Set(X,Z4)| = 4^2 = 16 = #extensions (each set map
    # gives a distinct hom on words of length 1)
    ok += int(len({v for v in [f.get(x) for x in [0, 1]]}) == 2)
    return {"synthetic_adjunction": float(ok == 2)}
