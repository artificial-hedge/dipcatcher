import pytest

from quant_fund.research import benches_w1862


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agravaine_qa_studies_family",
        "bench_isolde_qa_studies_family",
        "bench_kay_qa_studies_family",
        "bench_lyonesse_qa_studies_family",
        "bench_mark_cornwall_qa_studies_family",
        "bench_mordred_qa_studies_family",
    ],
)
def test_benches_w1862(fam):
    out = getattr(benches_w1862, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
