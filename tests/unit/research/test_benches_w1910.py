import pytest

from quant_fund.research import benches_w1910


@pytest.mark.parametrize(
    "fam",
    [
        "bench_demon_div_qa_studies_family",
        "bench_div_aq_qa_studies_family",
        "bench_div_demon_qa_studies_family",
        "bench_druj_spirit_qa_studies_family",
        "bench_nasu_demon_qa_studies_family",
        "bench_yalburz_qa_studies_family",
    ],
)
def test_benches_w1910(fam):
    out = getattr(benches_w1910, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
