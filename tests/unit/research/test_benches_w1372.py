import pytest

from quant_fund.research import benches_w1372


@pytest.mark.parametrize(
    "fam",
    [
        "bench_billsum_lite_studies_family",
        "bench_booksum_lite_studies_family",
        "bench_elm_lite_studies_family",
        "bench_govreport_lite_studies_family",
        "bench_qmsum_lite_studies_family",
        "bench_wikisum_lite_studies_family",
    ],
)
def test_benches_w1372(fam):
    out = getattr(benches_w1372, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
