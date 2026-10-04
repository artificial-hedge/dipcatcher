import pytest

from quant_fund.research import benches_w1384


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aider_polyglot_studies_family",
        "bench_hum_eval_studies_family",
        "bench_livebench_arena_studies_family",
        "bench_mbti_eval_studies_family",
        "bench_olmes_lite_studies_family",
        "bench_plus_eval_studies_family",
    ],
)
def test_benches_w1384(fam):
    out = getattr(benches_w1384, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
