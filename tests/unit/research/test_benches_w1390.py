import pytest

from quant_fund.research import benches_w1390


@pytest.mark.parametrize(
    "fam",
    [
        "bench_episum_lite_studies_family",
        "bench_fsum_lite_studies_family",
        "bench_mds_news_studies_family",
        "bench_sqcs_lite_studies_family",
        "bench_summon_fce_studies_family",
        "bench_wcep_lite_studies_family",
    ],
)
def test_benches_w1390(fam):
    out = getattr(benches_w1390, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
