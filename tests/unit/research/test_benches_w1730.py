import pytest

from quant_fund.research import benches_w1730


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aisit_qa_studies_family",
        "bench_bayna_qa_studies_family",
        "bench_kunkush_qa_studies_family",
        "bench_payna_qa_studies_family",
        "bench_taigan_qa_studies_family",
        "bench_yalyk_qa_studies_family",
    ],
)
def test_benches_w1730(fam):
    out = getattr(benches_w1730, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
