import pytest

from quant_fund.research import benches_w1399


@pytest.mark.parametrize(
    "fam",
    [
        "bench_asqa_lite_studies_family",
        "bench_eli5_lite_studies_family",
        "bench_fresh_qa_studies_family",
        "bench_nq_lite_studies_family",
        "bench_trivia_lite_studies_family",
        "bench_xor_tydi_studies_family",
    ],
)
def test_benches_w1399(fam):
    out = getattr(benches_w1399, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
