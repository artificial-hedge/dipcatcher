import pytest

from quant_fund.research import benches_w1263


@pytest.mark.parametrize(
    "fam",
    [
        "bench_diagnostic_meta_studies_family",
        "bench_fragility_index_studies_family",
        "bench_individual_patient_meta_studies_family",
        "bench_network_meta_studies_family",
        "bench_trial_sequential_studies_family",
        "bench_umbrella_review_studies_family",
    ],
)
def test_benches_w1263(fam):
    out = getattr(benches_w1263, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
