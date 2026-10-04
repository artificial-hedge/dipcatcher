import pytest

from quant_fund.research import benches_w1863


@pytest.mark.parametrize(
    "fam",
    [
        "bench_balin_qa_studies_family",
        "bench_lamorak_qa_studies_family",
        "bench_lot_qa_studies_family",
        "bench_marhaus_qa_studies_family",
        "bench_pellinor_qa_studies_family",
        "bench_uther_qa_studies_family",
    ],
)
def test_benches_w1863(fam):
    out = getattr(benches_w1863, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
