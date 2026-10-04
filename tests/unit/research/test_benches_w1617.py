import pytest

from quant_fund.research import benches_w1617


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amber_mountain_qa_studies_family",
        "bench_anosy_qa_studies_family",
        "bench_daraina_qa_studies_family",
        "bench_red_bellied_qa_studies_family",
        "bench_russet_qa_studies_family",
        "bench_white_footed_qa_studies_family",
    ],
)
def test_benches_w1617(fam):
    out = getattr(benches_w1617, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
