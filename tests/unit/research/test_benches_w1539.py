import pytest

from quant_fund.research import benches_w1539


@pytest.mark.parametrize(
    "fam",
    [
        "bench_goliath_heron_qa_studies_family",
        "bench_green_heron_qa_studies_family",
        "bench_grey_heron_qa_studies_family",
        "bench_night_heron_qa_studies_family",
        "bench_purple_heron_qa_studies_family",
        "bench_tiger_heron_qa_studies_family",
    ],
)
def test_benches_w1539(fam):
    out = getattr(benches_w1539, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
