import pytest

from quant_fund.research import benches_w1335


@pytest.mark.parametrize(
    "fam",
    [
        "bench_code_rag_studies_family",
        "bench_codegen_universal_studies_family",
        "bench_long_code_bench_studies_family",
        "bench_odex_eval_studies_family",
        "bench_swe_dev_studies_family",
        "bench_swe_multimodal_studies_family",
    ],
)
def test_benches_w1335(fam):
    out = getattr(benches_w1335, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
