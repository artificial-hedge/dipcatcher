import pytest

from quant_fund.research import benches_w1614


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hog_deer_qa_studies_family",
        "bench_kouprey_qa_studies_family",
        "bench_mule_qa_studies_family",
        "bench_pere_david_qa_studies_family",
        "bench_red_deer_qa_studies_family",
        "bench_wapiti_qa_studies_family",
    ],
)
def test_benches_w1614(fam):
    out = getattr(benches_w1614, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
