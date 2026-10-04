import pytest

from quant_fund.research import benches_w1790


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amun_qa_studies_family",
        "bench_atum_qa_studies_family",
        "bench_khepri_qa_studies_family",
        "bench_mut_qa_studies_family",
        "bench_ptah_qa_studies_family",
        "bench_seth_qa_studies_family",
    ],
)
def test_benches_w1790(fam):
    out = getattr(benches_w1790, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
