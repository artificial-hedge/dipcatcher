from quant_fund.models.graph_components import (
    articulation,
    bench_graph_components,
    bridges,
    tarjan_scc,
)


def test_scc_cycle():
    comp, nc = tarjan_scc(3, [(0, 1), (1, 2), (2, 0)])
    assert nc == 1
    assert len(set(comp.tolist())) == 1


def test_scc_two_cycles():
    edges = [(0, 1), (1, 0), (2, 3), (3, 2), (1, 2)]
    comp, nc = tarjan_scc(4, edges)
    assert nc == 2


def test_bridges_path():
    br = bridges(4, [(0, 1), (1, 2), (2, 3)])
    assert len(br) == 3


def test_no_bridge_in_cycle():
    br = bridges(3, [(0, 1), (1, 2), (2, 0)])
    assert len(br) == 0


def test_articulation_star():
    ap = articulation(4, [(0, 1), (0, 2), (0, 3)])
    assert ap[0] and not ap[1] and not ap[2]


def test_bench_keys():
    out = bench_graph_components()
    assert out["synthetic_scc_big"] == 2.0
    assert out["synthetic_articulation"] == 1.0
