import pytest

from quant_fund.research import benches_w1413


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anaphora_qa_studies_family",
        "bench_coherence_qa_studies_family",
        "bench_dialogue_act_studies_family",
        "bench_discourse_qa_studies_family",
        "bench_hedge_qa_studies_family",
        "bench_implicit_qa_studies_family",
    ],
)
def test_benches_w1413(fam):
    out = getattr(benches_w1413, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
