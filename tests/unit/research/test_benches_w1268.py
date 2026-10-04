import pytest

from quant_fund.research import benches_w1268


@pytest.mark.parametrize(
    "fam",
    [
        "bench_curiosity_diversity_studies_family",
        "bench_hindsight_relabel_studies_family",
        "bench_occupancy_measure_studies_family",
        "bench_option_discovery_studies_family",
        "bench_skill_chain_studies_family",
        "bench_successor_feature_studies_family",
    ],
)
def test_benches_w1268(fam):
    out = getattr(benches_w1268, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
