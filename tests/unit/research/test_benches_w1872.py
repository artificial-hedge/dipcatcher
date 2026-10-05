import pytest

from quant_fund.research import benches_w1872


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bugul_noz_qa_studies_family",
        "bench_cabyll_qa_studies_family",
        "bench_each_uisge_qa_studies_family",
        "bench_mooinjer_qa_studies_family",
        "bench_morveren_qa_studies_family",
        "bench_nuckelavee_qa_studies_family",
    ],
)
def test_benches_w1872(fam):
    out = getattr(benches_w1872, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
