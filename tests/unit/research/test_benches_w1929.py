import pytest

from quant_fund.research import benches_w1929


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hotupuku_qa_studies_family",
        "bench_kahui_tipua_qa_studies_family",
        "bench_kataore_qa_studies_family",
        "bench_nuku_mai_tore_qa_studies_family",
        "bench_tipua_qa_studies_family",
        "bench_wheke_muturangi_qa_studies_family",
    ],
)
def test_benches_w1929(fam):
    out = getattr(benches_w1929, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
