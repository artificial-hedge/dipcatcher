import pytest

from quant_fund.research import benches_w1734


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arawn_qa_studies_family",
        "bench_ceridwen_qa_studies_family",
        "bench_gwydion_qa_studies_family",
        "bench_llew_qa_studies_family",
        "bench_rhiannon_qa_studies_family",
        "bench_taliesin_qa_studies_family",
    ],
)
def test_benches_w1734(fam):
    out = getattr(benches_w1734, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
