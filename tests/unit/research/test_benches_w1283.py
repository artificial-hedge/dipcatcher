import pytest

from quant_fund.research import benches_w1283


@pytest.mark.parametrize(
    "fam",
    [
        "bench_context_compression_studies_family",
        "bench_episodic_memory_studies_family",
        "bench_memory_bank_studies_family",
        "bench_retrieval_memory_studies_family",
        "bench_semantic_memory_studies_family",
        "bench_working_memory_studies_family",
    ],
)
def test_benches_w1283(fam):
    out = getattr(benches_w1283, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
