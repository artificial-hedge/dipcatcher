import pytest

from quant_fund.research import benches_w1612


@pytest.mark.parametrize(
    "fam",
    [
        "bench_argali_qa_studies_family",
        "bench_bighorn_qa_studies_family",
        "bench_dall_qa_studies_family",
        "bench_llama_qa_studies_family",
        "bench_mouflon_qa_studies_family",
        "bench_urial_qa_studies_family",
    ],
)
def test_benches_w1612(fam):
    out = getattr(benches_w1612, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
