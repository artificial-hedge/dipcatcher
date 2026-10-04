import pytest

from quant_fund.research import benches_w1626


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bondolo_qa_studies_family",
        "bench_madame_berthe_qa_studies_family",
        "bench_mittermeier_qa_studies_family",
        "bench_northern_qa_studies_family",
        "bench_southern_qa_studies_family",
        "bench_western_qa_studies_family",
    ],
)
def test_benches_w1626(fam):
    out = getattr(benches_w1626, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
