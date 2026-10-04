import pytest

from quant_fund.research import benches_w1780


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arawn_qa_studies_family",
        "bench_branwen_qa_studies_family",
        "bench_gwydion_qa_studies_family",
        "bench_lleu_qa_studies_family",
        "bench_lludd_qa_studies_family",
        "bench_taliesin_qa_studies_family",
    ],
)
def test_benches_w1780(fam):
    out = getattr(benches_w1780, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
