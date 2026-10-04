import pytest

from quant_fund.research import benches_w1410


@pytest.mark.parametrize(
    "fam",
    [
        "bench_causal_qa_studies_family",
        "bench_ecare_lite_studies_family",
        "bench_event2mind_lite_studies_family",
        "bench_event_qa_studies_family",
        "bench_hippo_qa_studies_family",
        "bench_intent_qa_studies_family",
    ],
)
def test_benches_w1410(fam):
    out = getattr(benches_w1410, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
