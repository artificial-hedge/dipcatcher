import pytest

from quant_fund.research import benches_w1725


@pytest.mark.parametrize(
    "fam",
    [
        "bench_izanagi_qa_studies_family",
        "bench_izanami_qa_studies_family",
        "bench_kukunochi_qa_studies_family",
        "bench_omoikane_qa_studies_family",
        "bench_sarutahiko_qa_studies_family",
        "bench_uzume_qa_studies_family",
    ],
)
def test_benches_w1725(fam):
    out = getattr(benches_w1725, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
