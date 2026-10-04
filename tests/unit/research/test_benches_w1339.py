import pytest

from quant_fund.research import benches_w1339


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aqua_rat_studies_family",
        "bench_geo_qa_studies_family",
        "bench_hol_step_studies_family",
        "bench_math_odyssey_studies_family",
        "bench_tab_math_studies_family",
        "bench_uni_math_studies_family",
    ],
)
def test_benches_w1339(fam):
    out = getattr(benches_w1339, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
