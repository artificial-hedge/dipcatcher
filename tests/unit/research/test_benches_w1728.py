import pytest

from quant_fund.research import benches_w1728


@pytest.mark.parametrize(
    "fam",
    [
        "bench_boszorka_qa_studies_family",
        "bench_csaba_qa_studies_family",
        "bench_garabonci_qa_studies_family",
        "bench_isten_qa_studies_family",
        "bench_liderc_qa_studies_family",
        "bench_taltos_qa_studies_family",
    ],
)
def test_benches_w1728(fam):
    out = getattr(benches_w1728, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
