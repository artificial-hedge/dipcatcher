import pytest

from quant_fund.research import benches_w1765


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arianrhod_qa_studies_family",
        "bench_cerridwen_qa_studies_family",
        "bench_lugh_qa_studies_family",
        "bench_morrigan_qa_studies_family",
        "bench_nuada_qa_studies_family",
        "bench_rhiannon_qa_studies_family",
    ],
)
def test_benches_w1765(fam):
    out = getattr(benches_w1765, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
