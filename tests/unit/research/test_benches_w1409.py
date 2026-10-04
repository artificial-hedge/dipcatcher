import pytest

from quant_fund.research import benches_w1409


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bamboogle_lite_studies_family",
        "bench_beerqa_lite_studies_family",
        "bench_cider_qa_studies_family",
        "bench_ensem_qa_studies_family",
        "bench_fanqa_lite_studies_family",
        "bench_hops_qa_studies_family",
    ],
)
def test_benches_w1409(fam):
    out = getattr(benches_w1409, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
