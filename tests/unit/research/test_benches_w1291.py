import pytest

from quant_fund.research import benches_w1291


@pytest.mark.parametrize(
    "fam",
    [
        "bench_data_mix_studies_family",
        "bench_dedup_minhash_studies_family",
        "bench_dedup_studies_family",
        "bench_domain_classifier_studies_family",
        "bench_perplexity_filter_studies_family",
        "bench_quality_filter_studies_family",
    ],
)
def test_benches_w1291(fam):
    out = getattr(benches_w1291, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
