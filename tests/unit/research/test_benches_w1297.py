import pytest

from quant_fund.research import benches_w1297


@pytest.mark.parametrize(
    "fam",
    [
        "bench_benchmark_gaming_studies_family",
        "bench_benchmark_saturate_studies_family",
        "bench_contamination_studies_family",
        "bench_eval_coverage_studies_family",
        "bench_eval_reliability_studies_family",
        "bench_lm_eval_harness_studies_family",
    ],
)
def test_benches_w1297(fam):
    out = getattr(benches_w1297, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
