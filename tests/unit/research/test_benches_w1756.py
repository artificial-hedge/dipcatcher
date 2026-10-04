import pytest

from quant_fund.research import benches_w1756


@pytest.mark.parametrize(
    "fam",
    [
        "bench_apsara_qa_studies_family",
        "bench_gandharva_qa_studies_family",
        "bench_kinnara_qa_studies_family",
        "bench_ratri_qa_studies_family",
        "bench_rudra_qa_studies_family",
        "bench_ushas_qa_studies_family",
    ],
)
def test_benches_w1756(fam):
    out = getattr(benches_w1756, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
