import pytest

from quant_fund.research import benches_w1352


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bias_bench_studies_family",
        "bench_crowsp_lite_studies_family",
        "bench_honesty_lie_studies_family",
        "bench_social_iqa2_studies_family",
        "bench_stereo_lite_studies_family",
        "bench_wino_bias_studies_family",
    ],
)
def test_benches_w1352(fam):
    out = getattr(benches_w1352, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
