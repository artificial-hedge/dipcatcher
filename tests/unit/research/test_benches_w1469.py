import pytest

from quant_fund.research import benches_w1469


@pytest.mark.parametrize(
    "fam",
    [
        "bench_atoll_qa_studies_family",
        "bench_bluff_qa_studies_family",
        "bench_cove_qa_studies_family",
        "bench_headland_qa_studies_family",
        "bench_inlet_qa_studies_family",
        "bench_islet_qa_studies_family",
    ],
)
def test_benches_w1469(fam):
    out = getattr(benches_w1469, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
