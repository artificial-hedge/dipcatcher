"""Wave-277 adapter bench tests."""

from quant_fund.research.benches_w277 import (
    bench_chirp_z_family,
    bench_decimate_int_family,
    bench_fir_window_family,
    bench_prony_model_family,
    bench_stft_istft_family,
    bench_wola_synth_family,
)

FAMS = [
    bench_stft_istft_family,
    bench_chirp_z_family,
    bench_fir_window_family,
    bench_prony_model_family,
    bench_wola_synth_family,
    bench_decimate_int_family,
]


def test_wave277_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave277_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
