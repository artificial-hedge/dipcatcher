import pytest

from quant_fund.research import benches_w1570


@pytest.mark.parametrize(
    "fam",
    [
        "bench_fiddler_crab_qa_studies_family",
        "bench_ghost_crab_qa_studies_family",
        "bench_horseshoe_qa_studies_family",
        "bench_mud_crab_qa_studies_family",
        "bench_porcelain_qa_studies_family",
        "bench_spider_crab_qa_studies_family",
    ],
)
def test_benches_w1570(fam):
    out = getattr(benches_w1570, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
