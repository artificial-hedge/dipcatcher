from quant_fund.models.boolean_algebra import atomic, atoms, ba_laws, bench_boolean_algebra


def test_atoms_are_exactly_singletons():
    u = frozenset({0, 1, 2})
    assert atoms(u) == frozenset({frozenset({0}), frozenset({1}), frozenset({2})})
    # non-singletons and the empty set are not atoms
    assert frozenset({0, 1}) not in atoms(u)
    assert frozenset() not in atoms(u)


def test_empty_algebra():
    assert ba_laws(frozenset())
    assert atoms(frozenset()) == frozenset()
    # the trivial BA is (vacuously) atomic
    assert atomic(frozenset())


def test_atomic_powerset():
    assert atomic(frozenset({0, 1, 2}))


def test_bench_perfect():
    out = bench_boolean_algebra()
    assert out["synthetic_boolean_algebra"] == 1.0
