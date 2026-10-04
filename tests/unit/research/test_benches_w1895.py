import pytest

from quant_fund.research import benches_w1895


@pytest.mark.parametrize(
    "fam",
    [
        "bench_allatu_qa_studies_family",
        "bench_belili_qa_studies_family",
        "bench_dimme_qa_studies_family",
        "bench_gallu_qa_studies_family",
        "bench_lilu_qa_studies_family",
        "bench_sulak_qa_studies_family",
    ],
)
def test_benches_w1895(fam):
    out = getattr(benches_w1895, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
