import pytest

from quant_fund.research import benches_w1396


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agnews_lite_studies_family",
        "bench_dialsum_lite_studies_family",
        "bench_facet_lite_studies_family",
        "bench_medsum_lite_studies_family",
        "bench_oposum_lite_studies_family",
        "bench_qsum_lite_studies_family",
    ],
)
def test_benches_w1396(fam):
    out = getattr(benches_w1396, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
