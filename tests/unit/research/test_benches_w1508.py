import pytest

from quant_fund.research import benches_w1508


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chickadee_qa_studies_family",
        "bench_finch_qa_studies_family",
        "bench_sparrow_qa_studies_family",
        "bench_thrush_qa_studies_family",
        "bench_warbler_qa_studies_family",
        "bench_wren_qa_studies_family",
    ],
)
def test_benches_w1508(fam):
    out = getattr(benches_w1508, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
