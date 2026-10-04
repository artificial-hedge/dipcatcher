import pytest

from quant_fund.research import benches_w1269


@pytest.mark.parametrize(
    "fam",
    [
        "bench_best_of_n_studies_family",
        "bench_cdpo_studies_family",
        "bench_constitutional_ai_studies_family",
        "bench_orpo_studies_family",
        "bench_simpo_studies_family",
        "bench_sppo_studies_family",
    ],
)
def test_benches_w1269(fam):
    out = getattr(benches_w1269, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
