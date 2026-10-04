import pytest

from quant_fund.research import benches_w1736


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ilmarinen_qa_studies_family",
        "bench_joukahainen_qa_studies_family",
        "bench_lemminkainen_qa_studies_family",
        "bench_marjatta_qa_studies_family",
        "bench_tuoni_qa_studies_family",
        "bench_vainamoinen_qa_studies_family",
    ],
)
def test_benches_w1736(fam):
    out = getattr(benches_w1736, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
