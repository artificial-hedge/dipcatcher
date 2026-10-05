import pytest

from quant_fund.research import benches_w1827


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abaasy2_qa_studies_family",
        "bench_buga2_qa_studies_family",
        "bench_khosun2_qa_studies_family",
        "bench_kyys2_qa_studies_family",
        "bench_naa2_qa_studies_family",
        "bench_num2_qa_studies_family",
    ],
)
def test_benches_w1827(fam):
    out = getattr(benches_w1827, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
