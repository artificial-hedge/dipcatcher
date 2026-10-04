import pytest

from quant_fund.research import benches_w1445


@pytest.mark.parametrize(
    "fam",
    [
        "bench_birch_qa_studies_family",
        "bench_cedar_qa_studies_family",
        "bench_elm_qa_studies_family",
        "bench_maple_qa_studies_family",
        "bench_oak_qa_studies_family",
        "bench_willow_qa_studies_family",
    ],
)
def test_benches_w1445(fam):
    out = getattr(benches_w1445, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
