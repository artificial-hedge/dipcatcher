import pytest

from quant_fund.research import benches_w1391


@pytest.mark.parametrize(
    "fam",
    [
        "bench_archer_qa_studies_family",
        "bench_argue_eval_studies_family",
        "bench_expert_qa_studies_family",
        "bench_mintaka_lite_studies_family",
        "bench_musique_lite_studies_family",
        "bench_wiki2_qa_studies_family",
    ],
)
def test_benches_w1391(fam):
    out = getattr(benches_w1391, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
