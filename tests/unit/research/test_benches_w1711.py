import pytest

from quant_fund.research import benches_w1711


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baalat_qa_studies_family",
        "bench_dagon_qa_studies_family",
        "bench_eshmun_qa_studies_family",
        "bench_melqart_qa_studies_family",
        "bench_resheph_qa_studies_family",
        "bench_tanit_qa_studies_family",
    ],
)
def test_benches_w1711(fam):
    out = getattr(benches_w1711, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
