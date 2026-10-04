import pytest

from quant_fund.research import benches_w1436


@pytest.mark.parametrize(
    "fam",
    [
        "bench_comet_qa_studies_family",
        "bench_galaxy_qa_studies_family",
        "bench_moon_qa_studies_family",
        "bench_nebula_qa_studies_family",
        "bench_planet_qa_studies_family",
        "bench_star_qa_studies_family",
    ],
)
def test_benches_w1436(fam):
    out = getattr(benches_w1436, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
