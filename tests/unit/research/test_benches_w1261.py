import pytest

from quant_fund.research import benches_w1261


@pytest.mark.parametrize(
    "fam",
    [
        "bench_biostatistics_methods_studies_family",
        "bench_epidemiology_methods_studies_family",
        "bench_heor_studies_family",
        "bench_regulatory_science_studies_family",
        "bench_survival_trial_studies_family",
        "bench_translational_studies_family",
    ],
)
def test_benches_w1261(fam):
    out = getattr(benches_w1261, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
