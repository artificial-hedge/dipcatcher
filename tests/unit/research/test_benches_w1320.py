import pytest

from quant_fund.research import benches_w1320


@pytest.mark.parametrize(
    "fam",
    [
        "bench_gaia_bench_studies_family",
        "bench_mmbench_agent_studies_family",
        "bench_osworld_studies_family",
        "bench_screen_eval_studies_family",
        "bench_vsi_bench_studies_family",
        "bench_webvoyager_studies_family",
    ],
)
def test_benches_w1320(fam):
    out = getattr(benches_w1320, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
