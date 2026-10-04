import pytest

from quant_fund.research import benches_w1507


@pytest.mark.parametrize(
    "fam",
    [
        "bench_buzzard_qa_studies_family",
        "bench_caracara_qa_studies_family",
        "bench_goshawk_qa_studies_family",
        "bench_merlin_qa_studies_family",
        "bench_peregrine_qa_studies_family",
        "bench_sparrowhawk_qa_studies_family",
    ],
)
def test_benches_w1507(fam):
    out = getattr(benches_w1507, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
