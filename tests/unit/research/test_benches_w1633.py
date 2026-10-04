import pytest

from quant_fund.research import benches_w1633


@pytest.mark.parametrize(
    "fam",
    [
        "bench_air_sylph_qa_studies_family",
        "bench_earth_golem_qa_studies_family",
        "bench_fire_spirit_qa_studies_family",
        "bench_frost_wight_qa_studies_family",
        "bench_storm_jinn_qa_studies_family",
        "bench_water_sprite_qa_studies_family",
    ],
)
def test_benches_w1633(fam):
    out = getattr(benches_w1633, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
