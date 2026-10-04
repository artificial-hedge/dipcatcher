import pytest

from quant_fund.research import benches_w1325


@pytest.mark.parametrize(
    "fam",
    [
        "bench_babilong_studies_family",
        "bench_infinitebench_studies_family",
        "bench_longbench_studies_family",
        "bench_lv_eval_studies_family",
        "bench_ruler_bench_studies_family",
        "bench_zero_scrolls_studies_family",
    ],
)
def test_benches_w1325(fam):
    out = getattr(benches_w1325, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
