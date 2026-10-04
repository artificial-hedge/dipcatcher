import pytest

from quant_fund.research import benches_w1394


@pytest.mark.parametrize(
    "fam",
    [
        "bench_api_eval_studies_family",
        "bench_apps_lite_studies_family",
        "bench_livecode_studies_family",
        "bench_mbpp_lite_studies_family",
        "bench_restbench_studies_family",
        "bench_swe_gym_studies_family",
    ],
)
def test_benches_w1394(fam):
    out = getattr(benches_w1394, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
