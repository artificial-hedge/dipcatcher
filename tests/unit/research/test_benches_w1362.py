import pytest

from quant_fund.research import benches_w1362


@pytest.mark.parametrize(
    "fam",
    [
        "bench_check_that_studies_family",
        "bench_claim_buster_studies_family",
        "bench_emergent_lite_studies_family",
        "bench_fake_news_studies_family",
        "bench_snopes_lite_studies_family",
        "bench_stance_detect_studies_family",
    ],
)
def test_benches_w1362(fam):
    out = getattr(benches_w1362, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
