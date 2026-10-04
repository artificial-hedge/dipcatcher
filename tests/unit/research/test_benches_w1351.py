import pytest

from quant_fund.research import benches_w1351


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ethic_jiminy_studies_family",
        "bench_moral_exc_studies_family",
        "bench_moral_found_studies_family",
        "bench_principlism_toy_studies_family",
        "bench_scruples_lite_studies_family",
        "bench_virtue_ethics_studies_family",
    ],
)
def test_benches_w1351(fam):
    out = getattr(benches_w1351, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
