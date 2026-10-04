import pytest

from quant_fund.research import benches_w1363


@pytest.mark.parametrize(
    "fam",
    [
        "bench_age_bias_studies_family",
        "bench_curry_qa_studies_family",
        "bench_cw_qa2_studies_family",
        "bench_dialect_bias_studies_family",
        "bench_politi_fact_studies_family",
        "bench_rumor_twitter_studies_family",
    ],
)
def test_benches_w1363(fam):
    out = getattr(benches_w1363, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
