"""Wave-260 adapter tests."""

from quant_fund.research.benches_w260 import (
    bench_gale_chu_family,
    bench_gale_shapley_family,
    bench_hopcroft_karp_family,
    bench_konig_cover_family,
    bench_kuhn_munkres_family,
    bench_topo_layers_family,
)


def test_bench_gale_shapley_family():
    out = bench_gale_shapley_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_hopcroft_karp_family():
    out = bench_hopcroft_karp_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_kuhn_munkres_family():
    out = bench_kuhn_munkres_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_konig_cover_family():
    out = bench_konig_cover_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_gale_chu_family():
    out = bench_gale_chu_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_topo_layers_family():
    out = bench_topo_layers_family()
    assert out and all(k.startswith("synthetic_") for k in out)
