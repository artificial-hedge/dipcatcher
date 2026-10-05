import pytest

from quant_fund.research import benches_w1839


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alkutba2_qa_studies_family",
        "bench_aluzza2_qa_studies_family",
        "bench_dushara2_qa_studies_family",
        "bench_godil2_qa_studies_family",
        "bench_hubal2_qa_studies_family",
        "bench_manat2_qa_studies_family",
    ],
)
def test_benches_w1839(fam):
    out = getattr(benches_w1839, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
