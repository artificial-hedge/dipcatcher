import pytest

from quant_fund.research import benches_w1882


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ammed_qa_studies_family",
        "bench_laadas_qa_studies_family",
        "bench_langomed_qa_studies_family",
        "bench_mezzen_qa_studies_family",
        "bench_segimer_qa_studies_family",
        "bench_vercina_qa_studies_family",
    ],
)
def test_benches_w1882(fam):
    out = getattr(benches_w1882, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
