import pytest

from quant_fund.research import benches_w1344


@pytest.mark.parametrize(
    "fam",
    [
        "bench_boolq_qa_studies_family",
        "bench_dream_qa_studies_family",
        "bench_duorc_qa_studies_family",
        "bench_mctest_qa_studies_family",
        "bench_qasper_qa_studies_family",
        "bench_race_qa_studies_family",
    ],
)
def test_benches_w1344(fam):
    out = getattr(benches_w1344, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
