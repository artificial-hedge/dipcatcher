import pytest

from quant_fund.research import benches_w1433


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cloud_qa_studies_family",
        "bench_frost_qa_studies_family",
        "bench_hurricane_qa_studies_family",
        "bench_rain_qa_studies_family",
        "bench_storm_qa_studies_family",
        "bench_wind_qa_studies_family",
    ],
)
def test_benches_w1433(fam):
    out = getattr(benches_w1433, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
