import pytest

from quant_fund.research import benches_w1345


@pytest.mark.parametrize(
    "fam",
    [
        "bench_complex_qa_studies_family",
        "bench_entity_quests_studies_family",
        "bench_freebase_qa_studies_family",
        "bench_nq_open_studies_family",
        "bench_trivia_qa_studies_family",
        "bench_web_qa_studies_family",
    ],
)
def test_benches_w1345(fam):
    out = getattr(benches_w1345, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
