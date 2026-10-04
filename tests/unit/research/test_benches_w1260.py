import pytest

from quant_fund.research import benches_w1260


@pytest.mark.parametrize(
    "fam",
    [
        "bench_adaptive_trial_studies_family",
        "bench_clinical_trial_studies_family",
        "bench_comparative_effectiveness_studies_family",
        "bench_meta_analysis_studies_family",
        "bench_outcomes_research_studies_family",
        "bench_rwe_studies_family",
    ],
)
def test_benches_w1260(fam):
    out = getattr(benches_w1260, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
