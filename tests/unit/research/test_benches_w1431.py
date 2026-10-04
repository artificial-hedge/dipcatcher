import pytest

from quant_fund.research import benches_w1431


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aircraft_qa_studies_family",
        "bench_bike_qa_studies_family",
        "bench_bus_qa_studies_family",
        "bench_car_qa_studies_family",
        "bench_engine_qa_studies_family",
        "bench_plane_qa_studies_family",
    ],
)
def test_benches_w1431(fam):
    out = getattr(benches_w1431, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
