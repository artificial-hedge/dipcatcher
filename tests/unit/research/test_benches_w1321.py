import pytest

from quant_fund.research import benches_w1321


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chart_gqa_studies_family",
        "bench_mathvista_studies_family",
        "bench_mkqa_studies_family",
        "bench_mmmlu_studies_family",
        "bench_mmmu_studies_family",
        "bench_videomme_studies_family",
    ],
)
def test_benches_w1321(fam):
    out = getattr(benches_w1321, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
