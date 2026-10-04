import pytest

from quant_fund.research import benches_w1421


@pytest.mark.parametrize(
    "fam",
    [
        "bench_geospatial_qa_studies_family",
        "bench_itinerary_qa_studies_family",
        "bench_journey_qa_studies_family",
        "bench_route_qa_studies_family",
        "bench_spatial_qa_studies_family",
        "bench_terrain_qa_studies_family",
    ],
)
def test_benches_w1421(fam):
    out = getattr(benches_w1421, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
