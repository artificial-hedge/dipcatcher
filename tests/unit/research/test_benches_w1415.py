import pytest

from quant_fund.research import benches_w1415


@pytest.mark.parametrize(
    "fam",
    [
        "bench_case_qa_studies_family",
        "bench_clause_qa_studies_family",
        "bench_contract_qa_studies_family",
        "bench_lawqa_lite_studies_family",
        "bench_legal_qa_studies_family",
        "bench_statute_qa_studies_family",
    ],
)
def test_benches_w1415(fam):
    out = getattr(benches_w1415, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
