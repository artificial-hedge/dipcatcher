import pytest

from quant_fund.research import benches_w1502


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anteater_qa_studies_family",
        "bench_coatimundi_qa_studies_family",
        "bench_kinkajou_qa_studies_family",
        "bench_opossum_qa_studies_family",
        "bench_paca_qa_studies_family",
        "bench_tamandua_qa_studies_family",
    ],
)
def test_benches_w1502(fam):
    out = getattr(benches_w1502, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
