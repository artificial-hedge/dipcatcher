import pytest

from quant_fund.research import benches_w1290


@pytest.mark.parametrize(
    "fam",
    [
        "bench_awq_studies_family",
        "bench_entropy_code_quant_studies_family",
        "bench_gptq_studies_family",
        "bench_kv_cache_quant_studies_family",
        "bench_smoothquant_studies_family",
        "bench_weight_share_studies_family",
    ],
)
def test_benches_w1290(fam):
    out = getattr(benches_w1290, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
