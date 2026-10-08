"""EPaxos-lite ordering honesty tests."""

from __future__ import annotations

from quant_fund.models.epaxos import _apply, bench_epaxos, build_dag, topo_order


def test_topo_order_does_not_mutate_deps():
    cmds = [("a", 1), ("a", 2), ("b", 3), ("a", 4), ("b", 5)]
    deps = build_dag(cmds)
    snapshot = {i: set(d) for i, d in deps.items()}
    topo_order(deps)
    assert deps == snapshot


def test_topo_order_respects_conflicts():
    cmds = [("a", 1), ("b", 1), ("a", 2), ("a", 3), ("b", 2)]
    deps = build_dag(cmds)
    order = topo_order(deps)
    pos = {c: p for p, c in enumerate(order)}
    assert len(order) == len(cmds)
    for j, ds in deps.items():
        for i in ds:
            assert pos[i] < pos[j]


def test_different_extensions_same_state():
    # all ops on one key: every linear extension yields the same sum,
    # but order must still respect the chain i -> j for i < j
    cmds = [("k", 1), ("k", 2), ("k", 3), ("k", 4)]
    deps = build_dag(cmds)
    order = topo_order(deps)
    assert order == sorted(order)
    assert _apply({}, cmds, order) == {"k": 10}


def test_bench_epaxos_scores():
    out = bench_epaxos()
    assert out["synthetic_replicas_agree"] == 1.0
    assert out["synthetic_commutes_preserve_state"] == 1.0
    assert out["synthetic_deterministic_order"] == 1.0
