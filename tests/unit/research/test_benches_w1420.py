import pytest

from quant_fund.research import benches_w1420


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cause_qa_studies_family",
        "bench_claim_qa_studies_family",
        "bench_conclusion_qa_studies_family",
        "bench_deduction_qa_studies_family",
        "bench_effect_qa_studies_family",
        "bench_fallacy_qa_studies_family",
    ],
)
def test_benches_w1420(fam):
    out = getattr(benches_w1420, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
