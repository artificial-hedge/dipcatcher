import pytest

from quant_fund.research import benches_w1438


@pytest.mark.parametrize(
    "fam",
    [
        "bench_coral_qa_studies_family",
        "bench_dolphin_qa_studies_family",
        "bench_reef_qa_studies_family",
        "bench_shark_qa_studies_family",
        "bench_turtle_qa_studies_family",
        "bench_whale_qa_studies_family",
    ],
)
def test_benches_w1438(fam):
    out = getattr(benches_w1438, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
