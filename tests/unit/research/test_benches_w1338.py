import pytest

from quant_fund.research import benches_w1338


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arith_qa_studies_family",
        "bench_gsm_hard_studies_family",
        "bench_math_reason_studies_family",
        "bench_mini_f2f_studies_family",
        "bench_proof_pile_studies_family",
        "bench_theorem_qa_studies_family",
    ],
)
def test_benches_w1338(fam):
    out = getattr(benches_w1338, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
