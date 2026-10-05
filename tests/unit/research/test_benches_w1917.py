import pytest

from quant_fund.research import benches_w1917


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baobhan_sith_qa_studies_family",
        "bench_bean_nighe_qa_studies_family",
        "bench_boggart_qa_studies_family",
        "bench_fear_durach_qa_studies_family",
        "bench_glaistig_qa_studies_family",
        "bench_sluagh_qa_studies_family",
    ],
)
def test_benches_w1917(fam):
    out = getattr(benches_w1917, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
