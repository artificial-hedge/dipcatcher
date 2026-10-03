"""Adapter tests for wave-303 speech/audio codec canon benches."""

from quant_fund.research.benches_w303 import (
    bench_adpcm_ima_family,
    bench_celp_encode_family,
    bench_lpc_analysis_family,
    bench_mel_cepstrum_family,
    bench_mulaw_compand_family,
    bench_viterbi_vad_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_mulaw_compand_family,
        bench_adpcm_ima_family,
        bench_lpc_analysis_family,
        bench_celp_encode_family,
        bench_mel_cepstrum_family,
        bench_viterbi_vad_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
