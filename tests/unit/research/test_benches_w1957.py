import pytest

from quant_fund.research import benches_w1957


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bael_qa_studies_family",
        "bench_bifrons_qa_studies_family",
        "bench_crocell_qa_studies_family",
        "bench_gamigin_qa_studies_family",
        "bench_haagenti_qa_studies_family",
        "bench_vuall_qa_studies_family",
    ],
)
def test_benches_w1957(fam):
    out = getattr(benches_w1957, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
