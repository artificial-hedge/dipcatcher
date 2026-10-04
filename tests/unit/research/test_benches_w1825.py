import pytest

from quant_fund.research import benches_w1825


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aitvaras2_qa_studies_family",
        "bench_kaukas2_qa_studies_family",
        "bench_laime2_qa_studies_family",
        "bench_perkunas2_qa_studies_family",
        "bench_velnias2_qa_studies_family",
        "bench_zemyna2_qa_studies_family",
    ],
)
def test_benches_w1825(fam):
    out = getattr(benches_w1825, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
