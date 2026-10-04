import pytest

from quant_fund.research import benches_w1293


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cai_critique_studies_family",
        "bench_constitutional_studies_family",
        "bench_harmlessness_rl_studies_family",
        "bench_principle_eval_studies_family",
        "bench_rlaif_studies_family",
        "bench_sleeper_eval_studies_family",
    ],
)
def test_benches_w1293(fam):
    out = getattr(benches_w1293, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
