"""Wave-273 adapter bench tests."""

from quant_fund.research.benches_w273 import (
    bench_const_fold_family,
    bench_inline_expand_family,
    bench_loop_unroll_family,
    bench_partial_eval_family,
    bench_peephole_opt_family,
    bench_strength_red_family,
)

FAMS = [
    bench_partial_eval_family,
    bench_peephole_opt_family,
    bench_strength_red_family,
    bench_const_fold_family,
    bench_loop_unroll_family,
    bench_inline_expand_family,
]


def test_wave273_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave273_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
