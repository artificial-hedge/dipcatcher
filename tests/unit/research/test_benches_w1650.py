import pytest

from quant_fund.research import benches_w1650


@pytest.mark.parametrize(
    "fam",
    [
        "bench_adjule_qa_studies_family",
        "bench_agogwe_qa_studies_family",
        "bench_biloko_qa_studies_family",
        "bench_kongamato_qa_studies_family",
        "bench_popobawa_qa_studies_family",
        "bench_rompo_qa_studies_family",
    ],
)
def test_benches_w1650(fam):
    out = getattr(benches_w1650, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
