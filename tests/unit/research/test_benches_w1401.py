import pytest

from quant_fund.research import benches_w1401


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alg514_lite_studies_family",
        "bench_dolphin_lite_studies_family",
        "bench_draw_lite_studies_family",
        "bench_lila_lite_studies_family",
        "bench_math_doc_studies_family",
        "bench_math_eval_studies_family",
    ],
)
def test_benches_w1401(fam):
    out = getattr(benches_w1401, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
