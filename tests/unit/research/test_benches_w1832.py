import pytest

from quant_fund.research import benches_w1832


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baalat2_qa_studies_family",
        "bench_eshmun2_qa_studies_family",
        "bench_melqart2_qa_studies_family",
        "bench_reshef2_qa_studies_family",
        "bench_tanit2_qa_studies_family",
        "bench_yam2_qa_studies_family",
    ],
)
def test_benches_w1832(fam):
    out = getattr(benches_w1832, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
