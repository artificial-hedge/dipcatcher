import pytest

from quant_fund.research import benches_w1407


@pytest.mark.parametrize(
    "fam",
    [
        "bench_canard_lite_studies_family",
        "bench_clarq_lite_studies_family",
        "bench_doqa_lite_studies_family",
        "bench_duread_qa_studies_family",
        "bench_orchid_qa_studies_family",
        "bench_qrecc_lite_studies_family",
    ],
)
def test_benches_w1407(fam):
    out = getattr(benches_w1407, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
