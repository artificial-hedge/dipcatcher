import pytest

from quant_fund.research import benches_w1742


@pytest.mark.parametrize(
    "fam",
    [
        "bench_haumia_qa_studies_family",
        "bench_rongo_qa_studies_family",
        "bench_tane_qa_studies_family",
        "bench_tangaroa_qa_studies_family",
        "bench_tawhirimatea_qa_studies_family",
        "bench_tumatauenga_qa_studies_family",
    ],
)
def test_benches_w1742(fam):
    out = getattr(benches_w1742, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
