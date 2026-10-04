import pytest

from quant_fund.research import benches_w1702


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chac_qa_studies_family",
        "bench_hunab_qa_studies_family",
        "bench_itzamna_qa_studies_family",
        "bench_ixchel_qa_studies_family",
        "bench_kukulcan_qa_studies_family",
        "bench_yumkaax_qa_studies_family",
    ],
)
def test_benches_w1702(fam):
    out = getattr(benches_w1702, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
