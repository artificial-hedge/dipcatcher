"""Wave-274 adapter bench tests."""

from quant_fund.research.benches_w274 import (
    bench_gc_skew_family,
    bench_hmm_profile_family,
    bench_kmer_count_family,
    bench_orf_find_family,
    bench_seq_logo_family,
    bench_star_msa_family,
)

FAMS = [
    bench_hmm_profile_family,
    bench_star_msa_family,
    bench_gc_skew_family,
    bench_orf_find_family,
    bench_kmer_count_family,
    bench_seq_logo_family,
]


def test_wave274_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave274_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
