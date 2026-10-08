from quant_fund.models.boundary_sq import _boundary, bench_boundary_sq


def test_triangle_boundary_has_three_edges():
    """A k-simplex's boundary has k+1 faces — 3 for a triangle. The old
    bench asserted even column sums at every level, which is vacuously
    false for triangles; the real property is d1 @ d2 = 0."""
    edges = [(0, 1), (0, 2), (1, 2)]
    tris = [(0, 1, 2)]
    d2 = _boundary(tris, edges)
    assert (d2.sum(axis=0) == 3).all()
    verts = [(0,), (1,), (2,)]
    d1 = _boundary(edges, verts)
    assert ((d1 @ d2) % 2 == 0).all()


def test_edge_boundary_is_two_vertices():
    edges = [(0, 1)]
    verts = [(0,), (1,)]
    d1 = _boundary(edges, verts)
    assert (d1.sum(axis=0) == 2).all()


def test_empty_faces_make_vacuous_product():
    # an empty face list yields a zero-column matrix; a composition
    # against it is vacuously zero — the bench must not count that as
    # a real ∂²=0 verification
    d_empty = _boundary([], [(0,), (1,)])
    assert d_empty.shape == (2, 0)


def test_bench_verifies_composition_on_triangle_sample():
    out = bench_boundary_sq()
    assert out["synthetic_boundary_sq"] == 1.0
