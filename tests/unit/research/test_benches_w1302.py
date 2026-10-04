import pytest

from quant_fund.research import benches_w1302


@pytest.mark.parametrize(
    "fam",
    [
        "bench_canary_infer_studies_family",
        "bench_deep_leak_studies_family",
        "bench_gradient_leak_studies_family",
        "bench_lira_studies_family",
        "bench_membership_infer_studies_family",
        "bench_shadow_model_studies_family",
    ],
)
def test_benches_w1302(fam):
    out = getattr(benches_w1302, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
