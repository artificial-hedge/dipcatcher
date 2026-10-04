import pytest

from quant_fund.research import benches_w1371


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ethos_lite_studies_family",
        "bench_moral_stories_studies_family",
        "bench_mutual_lite_studies_family",
        "bench_prosocial_lite_studies_family",
        "bench_scruples_studies_family",
        "bench_siqa_lite_studies_family",
    ],
)
def test_benches_w1371(fam):
    out = getattr(benches_w1371, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
