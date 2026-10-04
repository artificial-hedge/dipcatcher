import pytest

from quant_fund.research import benches_w1506


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bufflehead_qa_studies_family",
        "bench_canvasback_qa_studies_family",
        "bench_eider_qa_studies_family",
        "bench_mallard_qa_studies_family",
        "bench_merganser_qa_studies_family",
        "bench_scoter_qa_studies_family",
    ],
)
def test_benches_w1506(fam):
    out = getattr(benches_w1506, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
