import pytest

from quant_fund.research import benches_w1925


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abaasy_qa_studies_family",
        "bench_chedipe_qa_studies_family",
        "bench_kus_qa_studies_family",
        "bench_kyys_qa_studies_family",
        "bench_oror_qa_studies_family",
        "bench_urgut_qa_studies_family",
    ],
)
def test_benches_w1925(fam):
    out = getattr(benches_w1925, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
