"""Wave-333 computer-algebra-2 adapter tests."""

from quant_fund.research.benches_w333 import (
    bench_hensel_lift_family,
    bench_poly_crt_family,
    bench_poly_eval_interp_family,
    bench_poly_factor_fp_family,
    bench_sparse_interp_family,
    bench_subresultant_family,
)


def test_poly_factor_fp_family():
    out = bench_poly_factor_fp_family()
    assert out["synthetic_poly_factor_fp"] == 1.0


def test_hensel_lift_family():
    out = bench_hensel_lift_family()
    assert out["synthetic_hensel_lift"] == 1.0


def test_poly_crt_family():
    out = bench_poly_crt_family()
    assert out["synthetic_poly_crt"] == 1.0


def test_subresultant_family():
    out = bench_subresultant_family()
    assert out["synthetic_subresultant"] == 1.0


def test_sparse_interp_family():
    out = bench_sparse_interp_family()
    assert out["synthetic_sparse_interp"] == 1.0


def test_poly_eval_interp_family():
    out = bench_poly_eval_interp_family()
    assert out["synthetic_poly_eval_interp"] == 1.0
