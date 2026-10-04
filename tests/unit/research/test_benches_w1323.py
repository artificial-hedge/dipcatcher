import pytest

from quant_fund.research import benches_w1323


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bigcodebench_studies_family",
        "bench_ds1000_studies_family",
        "bench_humaneval_plus_studies_family",
        "bench_livecodebench_studies_family",
        "bench_mbpp_plus_studies_family",
        "bench_swe_perf_studies_family",
    ],
)
def test_benches_w1323(fam):
    out = getattr(benches_w1323, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
