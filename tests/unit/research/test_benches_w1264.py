import pytest

from quant_fund.research import benches_w1264


@pytest.mark.parametrize(
    "fam",
    [
        "bench_external_control_studies_family",
        "bench_negative_control_studies_family",
        "bench_probabilistic_bias_studies_family",
        "bench_self_controlled_studies_family",
        "bench_structural_nested_studies_family",
        "bench_transportability_studies_family",
    ],
)
def test_benches_w1264(fam):
    out = getattr(benches_w1264, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
