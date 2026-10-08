from quant_fund.models.cascades_opt import _cost, bench_cascades_opt, optimize


def test_equal_card_canonical_plan() -> None:
    # all-equal cardinalities: every complete plan costs the same, so the
    # winner is whichever candidate is enumerated FIRST — that order must
    # be canonical (sorted), not frozenset/hash order (PYTHONHASHSEED).
    tables = ["t0", "t1", "t2", "t3"]
    cards = {t: 100.0 for t in tables}
    plan = optimize(tables, cards)
    # sorted enumeration reaches the (t0,t1)|(t2,t3) balanced split first
    assert plan == (
        "hj",
        ("hj", ("scan", "t0"), ("scan", "t1")),
        ("hj", ("scan", "t2"), ("scan", "t3")),
    )


def test_optimal_beats_left_deep() -> None:
    tables = ["t0", "t1", "t2", "t3"]
    cards = {"t0": 10.0, "t1": 9000.0, "t2": 20.0, "t3": 15.0}
    plan = optimize(tables, cards)
    assert _cost(plan, cards) < _cost(
        ("nlj", ("nlj", ("nlj", ("scan", "t0"), ("scan", "t1")), ("scan", "t2")), ("scan", "t3")),
        cards,
    )


def test_deterministic_repeat() -> None:
    tables = ["a", "b", "c", "d"]
    cards = {t: float(i + 1) * 37.0 for i, t in enumerate(tables)}
    assert optimize(tables, cards) == optimize(tables, cards)


def test_bench_passes() -> None:
    assert bench_cascades_opt()["synthetic_cascades_optimal"] == 1.0
