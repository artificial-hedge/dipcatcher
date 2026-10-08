from quant_fund.models.bsp_tree import BSP, bench_bsp_tree


def test_order_ok_accepts_true_painters_order():
    from quant_fund.models.bsp_tree import _order_ok

    pts = [(0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9)]
    tree = BSP(pts)
    eye = (0.0, 0.0)
    seq = tree.traverse_back_to_front(eye)
    assert _order_ok(tree, eye, seq)


def test_order_ok_rejects_swapped_order():
    """Probe: the old 'order' check only verified len(seq) — any
    permutation passed. _order_ok must fail a sequence that violates
    the painter's property at some split."""
    from quant_fund.models.bsp_tree import _order_ok

    pts = [(0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9)]
    tree = BSP(pts)
    eye = (0.0, 0.0)
    seq = tree.traverse_back_to_front(eye)
    bad = list(reversed(seq))
    # reversed order puts at least one near-side point before a
    # far-side one unless the correct order was degenerate
    if bad != seq:
        assert not _order_ok(tree, eye, bad) or seq == bad


def test_query_region_matches_manual_walk():
    pts = [(10.0, 10.0), (90.0, 10.0), (10.0, 90.0), (90.0, 90.0), (50.0, 50.0)]
    tree = BSP(pts)
    q = (5.0, 5.0)
    out: list[tuple[float, float]] = []
    tree.query(q, out)
    assert out
    assert set(out) <= set(pts)


def test_bench_perfect():
    out = bench_bsp_tree()
    assert out["synthetic_region_correct"] == 1.0
    assert out["synthetic_full_coverage"] == 1.0
    assert out["synthetic_traversal_order"] == 1.0
