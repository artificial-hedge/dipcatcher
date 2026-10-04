import pytest

from quant_fund.research import benches_w1327


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bbq_bias_studies_family",
        "bench_bold_bias_studies_family",
        "bench_crowspairs_studies_family",
        "bench_holist_bias_studies_family",
        "bench_realtoxicity_studies_family",
        "bench_toxigen_eval_studies_family",
    ],
)
def test_benches_w1327(fam):
    out = getattr(benches_w1327, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
