import pytest

from quant_fund.research import benches_w1265


@pytest.mark.parametrize(
    "fam",
    [
        "bench_colocalization_studies_family",
        "bench_genetic_correlation_studies_family",
        "bench_heritability_ldscore_studies_family",
        "bench_mendelian_randomization_studies_family",
        "bench_pleiotropy_robust_studies_family",
        "bench_polygenic_score_studies_family",
    ],
)
def test_benches_w1265(fam):
    out = getattr(benches_w1265, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
