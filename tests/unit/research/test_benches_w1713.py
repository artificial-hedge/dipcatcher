import pytest

from quant_fund.research import benches_w1713


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amirani_qa_studies_family",
        "bench_apsat_qa_studies_family",
        "bench_barbale_qa_studies_family",
        "bench_dalis_qa_studies_family",
        "bench_ghmerti_qa_studies_family",
        "bench_kamar_qa_studies_family",
    ],
)
def test_benches_w1713(fam):
    out = getattr(benches_w1713, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
