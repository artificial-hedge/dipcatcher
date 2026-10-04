import pytest

from quant_fund.research import benches_w1411


@pytest.mark.parametrize(
    "fam",
    [
        "bench_fakeqa_lite_studies_family",
        "bench_flame_qa_studies_family",
        "bench_hate_qa_studies_family",
        "bench_ironic_qa_studies_family",
        "bench_offensive_qa_studies_family",
        "bench_politeness_qa_studies_family",
    ],
)
def test_benches_w1411(fam):
    out = getattr(benches_w1411, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
