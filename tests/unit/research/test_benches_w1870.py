import pytest

from quant_fund.research import benches_w1870


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bucca_qa_studies_family",
        "bench_knocker_qa_studies_family",
        "bench_morgawr_qa_studies_family",
        "bench_piskie_qa_studies_family",
        "bench_spriggan_qa_studies_family",
        "bench_tregeagle_qa_studies_family",
    ],
)
def test_benches_w1870(fam):
    out = getattr(benches_w1870, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
