import pytest

from quant_fund.research import benches_w1567


@pytest.mark.parametrize(
    "fam",
    [
        "bench_barbel_qa_studies_family",
        "bench_bream_qa_studies_family",
        "bench_carp_qa_studies_family",
        "bench_minnow_qa_studies_family",
        "bench_roach_qa_studies_family",
        "bench_tench_qa_studies_family",
    ],
)
def test_benches_w1567(fam):
    out = getattr(benches_w1567, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
