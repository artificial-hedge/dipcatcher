import pytest

from quant_fund.research import benches_w1280


@pytest.mark.parametrize(
    "fam",
    [
        "bench_data_mixture_studies_family",
        "bench_data_quality_studies_family",
        "bench_dedup_pipeline_studies_family",
        "bench_domain_filtering_studies_family",
        "bench_synthetic_data_studies_family",
        "bench_token_budget_studies_family",
    ],
)
def test_benches_w1280(fam):
    out = getattr(benches_w1280, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
