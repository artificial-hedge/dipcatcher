import pytest

from quant_fund.research import benches_w1497


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bettong_qa_studies_family",
        "bench_cuscus_qa_studies_family",
        "bench_numbat2_qa_studies_family",
        "bench_pademelon_qa_studies_family",
        "bench_potoroo_qa_studies_family",
        "bench_woylie_qa_studies_family",
    ],
)
def test_benches_w1497(fam):
    out = getattr(benches_w1497, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
