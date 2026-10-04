import pytest

from quant_fund.research import benches_w1564


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bobtail_squid_qa_studies_family",
        "bench_cuttlefish_qa_studies_family",
        "bench_nautilus_qa_studies_family",
        "bench_nudibranch_qa_studies_family",
        "bench_sea_slug_qa_studies_family",
        "bench_vampire_squid_qa_studies_family",
    ],
)
def test_benches_w1564(fam):
    out = getattr(benches_w1564, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
