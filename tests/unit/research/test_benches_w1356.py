import pytest

from quant_fund.research import benches_w1356


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bigbench_lite_studies_family",
        "bench_entity_qa_studies_family",
        "bench_mmlu_lite_studies_family",
        "bench_natural_qa_studies_family",
        "bench_pop_qa_studies_family",
        "bench_triviaqa_lite_studies_family",
    ],
)
def test_benches_w1356(fam):
    out = getattr(benches_w1356, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
