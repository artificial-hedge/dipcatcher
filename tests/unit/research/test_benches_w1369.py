import pytest

from quant_fund.research import benches_w1369


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aime_eval_studies_family",
        "bench_asdiv_lite_studies_family",
        "bench_math500_lite_studies_family",
        "bench_mgsm_lite_studies_family",
        "bench_minerva_math_studies_family",
        "bench_svamp_lite_studies_family",
    ],
)
def test_benches_w1369(fam):
    out = getattr(benches_w1369, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
