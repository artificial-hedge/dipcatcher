import pytest

from quant_fund.research import benches_w1386


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alfworld_lite_studies_family",
        "bench_babyai_lite_studies_family",
        "bench_crafter_lite_studies_family",
        "bench_jericho_lite_studies_family",
        "bench_scienceworld_studies_family",
        "bench_textworld_lite_studies_family",
    ],
)
def test_benches_w1386(fam):
    out = getattr(benches_w1386, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
