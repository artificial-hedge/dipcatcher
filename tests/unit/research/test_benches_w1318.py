import pytest

from quant_fund.research import benches_w1318


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cb_studies_family",
        "bench_cola_studies_family",
        "bench_qqp_studies_family",
        "bench_squad_v2_studies_family",
        "bench_sst2_studies_family",
        "bench_wic_studies_family",
    ],
)
def test_benches_w1318(fam):
    out = getattr(benches_w1318, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
