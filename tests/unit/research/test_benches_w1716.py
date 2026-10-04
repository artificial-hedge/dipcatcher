import pytest

from quant_fund.research import benches_w1716


@pytest.mark.parametrize(
    "fam",
    [
        "bench_heroas_qa_studies_family",
        "bench_kottiso_qa_studies_family",
        "bench_kotys_qa_studies_family",
        "bench_semele_qa_studies_family",
        "bench_theandrites_qa_studies_family",
        "bench_zibelthiurdos_qa_studies_family",
    ],
)
def test_benches_w1716(fam):
    out = getattr(benches_w1716, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
