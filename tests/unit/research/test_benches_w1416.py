import pytest

from quant_fund.research import benches_w1416


@pytest.mark.parametrize(
    "fam",
    [
        "bench_analyst_qa_studies_family",
        "bench_audit_qa_studies_family",
        "bench_bank_qa_studies_family",
        "bench_broker_qa_studies_family",
        "bench_credit_qa_studies_family",
        "bench_earnings_qa_studies_family",
    ],
)
def test_benches_w1416(fam):
    out = getattr(benches_w1416, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
