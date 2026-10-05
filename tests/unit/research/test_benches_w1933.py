import pytest

from quant_fund.research import benches_w1933


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bacoo_qa_studies_family",
        "bench_duppy_qa_studies_family",
        "bench_jumbie_qa_studies_family",
        "bench_lagahoo_qa_studies_family",
        "bench_ole_higue_qa_studies_family",
        "bench_soucouyant_qa_studies_family",
    ],
)
def test_benches_w1933(fam):
    out = getattr(benches_w1933, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
