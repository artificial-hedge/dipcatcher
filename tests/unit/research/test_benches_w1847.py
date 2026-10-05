import pytest

from quant_fund.research import benches_w1847


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dhatanwat_qa_studies_family",
        "bench_dhatzahran_qa_studies_family",
        "bench_hawl_qa_studies_family",
        "bench_khalasah_qa_studies_family",
        "bench_raymah_qa_studies_family",
        "bench_shams_qa_studies_family",
    ],
)
def test_benches_w1847(fam):
    out = getattr(benches_w1847, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
