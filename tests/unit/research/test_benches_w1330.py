import pytest

from quant_fund.research import benches_w1330


@pytest.mark.parametrize(
    "fam",
    [
        "bench_beaver_safe_studies_family",
        "bench_do_not_answer_studies_family",
        "bench_hh_rlhf_studies_family",
        "bench_honest_eval_studies_family",
        "bench_safe_rlhf_studies_family",
        "bench_sos_bench_studies_family",
    ],
)
def test_benches_w1330(fam):
    out = getattr(benches_w1330, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
