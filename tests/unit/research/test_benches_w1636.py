import pytest

from quant_fund.research import benches_w1636


@pytest.mark.parametrize(
    "fam",
    [
        "bench_gashadokuro_qa_studies_family",
        "bench_jorogumo_qa_studies_family",
        "bench_kodama_qa_studies_family",
        "bench_namahage_qa_studies_family",
        "bench_nue_2_qa_studies_family",
        "bench_tsuchinoko_qa_studies_family",
    ],
)
def test_benches_w1636(fam):
    out = getattr(benches_w1636, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
