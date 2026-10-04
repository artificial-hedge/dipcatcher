import pytest

from quant_fund.research import benches_w1326


@pytest.mark.parametrize(
    "fam",
    [
        "bench_attribute_inference_studies_family",
        "bench_canary_memorization_studies_family",
        "bench_extraction_attack_studies_family",
        "bench_membership_inference_studies_family",
        "bench_model_inversion_studies_family",
        "bench_privacy_meter_studies_family",
    ],
)
def test_benches_w1326(fam):
    out = getattr(benches_w1326, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
