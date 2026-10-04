import pytest

from quant_fund.research import benches_w1295


@pytest.mark.parametrize(
    "fam",
    [
        "bench_activation_patch_studies_family",
        "bench_circuit_tracer_studies_family",
        "bench_feature_dashboard_studies_family",
        "bench_jailbreak_detect_studies_family",
        "bench_mech_anomaly_studies_family",
        "bench_sae_linter_studies_family",
    ],
)
def test_benches_w1295(fam):
    out = getattr(benches_w1295, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
