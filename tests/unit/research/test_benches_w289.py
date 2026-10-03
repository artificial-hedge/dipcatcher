"""Adapter tests for wave-289 category-theory canon benches."""

from quant_fund.research.benches_w289 import (
    bench_adjunction_family,
    bench_fin_cat_family,
    bench_functor_check_family,
    bench_limit_prod_family,
    bench_monad_laws_family,
    bench_nat_trans_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_fin_cat_family,
        bench_functor_check_family,
        bench_nat_trans_family,
        bench_adjunction_family,
        bench_limit_prod_family,
        bench_monad_laws_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
