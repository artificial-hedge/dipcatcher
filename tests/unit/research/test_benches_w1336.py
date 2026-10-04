import pytest

from quant_fund.research import benches_w1336


@pytest.mark.parametrize(
    "fam",
    [
        "bench_assistantbench_studies_family",
        "bench_mind2web_studies_family",
        "bench_miniwob_studies_family",
        "bench_visual_web_studies_family",
        "bench_web_nav_studies_family",
        "bench_webarena_studies_family",
    ],
)
def test_benches_w1336(fam):
    out = getattr(benches_w1336, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
