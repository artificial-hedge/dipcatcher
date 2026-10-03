"""Wave-280 adapter bench tests."""

from quant_fund.research.benches_w280 import (
    bench_boundary_sq_family,
    bench_euler_char_family,
    bench_graph_h1_family,
    bench_rips_h1_family,
    bench_simp_betti_family,
    bench_winding_deg_family,
)

FAMS = [
    bench_simp_betti_family,
    bench_boundary_sq_family,
    bench_euler_char_family,
    bench_rips_h1_family,
    bench_graph_h1_family,
    bench_winding_deg_family,
]


def test_wave280_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave280_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
