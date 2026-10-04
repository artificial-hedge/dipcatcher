import pytest

from quant_fund.research import benches_w1412


@pytest.mark.parametrize(
    "fam",
    [
        "bench_affect_qa_studies_family",
        "bench_anger_qa_studies_family",
        "bench_comfort_qa_studies_family",
        "bench_distress_qa_studies_family",
        "bench_emotion_qa_studies_family",
        "bench_empathy_qa_studies_family",
    ],
)
def test_benches_w1412(fam):
    out = getattr(benches_w1412, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
