from quant_fund.models.association_rules import (
    apriori,
    bench_association_rules,
    mine_rules,
)


def test_apriori_finds_frequent():
    tx = [{1, 2}, {1, 2, 3}, {1, 2}, {3}, {1, 2, 3}, {1, 2}]
    sup = apriori(tx, min_support=0.4)
    assert frozenset({1, 2}) in sup
    assert abs(sup[frozenset({1, 2})] - 5 / 6) < 1e-9


def test_mine_rules_conf_lift():
    sup = {frozenset({1}): 0.6, frozenset({2}): 0.5, frozenset({1, 2}): 0.5}
    rules = mine_rules(sup, min_conf=0.5)
    r = [x for x in rules if x["ante"] == frozenset({1})][0]
    assert abs(r["confidence"] - 0.5 / 0.6) < 1e-9
    assert r["lift"] > 1.0


def test_bench_association_rules():
    out = bench_association_rules(seed=568)
    assert out["synthetic_rule_conf"] > 0.75
    assert out["synthetic_spurious_rules"] == 0
