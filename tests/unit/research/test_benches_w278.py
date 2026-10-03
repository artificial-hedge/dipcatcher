"""Wave-278 adapter bench tests."""

from quant_fund.research.benches_w278 import (
    bench_cobweb_model_family,
    bench_nk_phillips_family,
    bench_olg_model_family,
    bench_rbc_sim_family,
    bench_solow_model_family,
    bench_taylor_rule_family,
)

FAMS = [
    bench_rbc_sim_family,
    bench_nk_phillips_family,
    bench_taylor_rule_family,
    bench_solow_model_family,
    bench_olg_model_family,
    bench_cobweb_model_family,
]


def test_wave278_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave278_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
