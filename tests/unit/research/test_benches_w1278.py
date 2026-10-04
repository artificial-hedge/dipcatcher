import pytest

from quant_fund.research import benches_w1278


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chunked_prefill_studies_family",
        "bench_continuous_batching_studies_family",
        "bench_disaggregated_serving_studies_family",
        "bench_early_exit_studies_family",
        "bench_prefix_caching_studies_family",
        "bench_tensor_parallel_studies_family",
    ],
)
def test_benches_w1278(fam):
    out = getattr(benches_w1278, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
