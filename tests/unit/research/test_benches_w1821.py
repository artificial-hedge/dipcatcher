import pytest

from quant_fund.research import benches_w1821


@pytest.mark.parametrize(
    "fam",
    [
        "bench_benzaiten2_qa_studies_family",
        "bench_daikoku2_qa_studies_family",
        "bench_ebisu2_qa_studies_family",
        "bench_fukurokuju2_qa_studies_family",
        "bench_hotei2_qa_studies_family",
        "bench_juroujin2_qa_studies_family",
    ],
)
def test_benches_w1821(fam):
    out = getattr(benches_w1821, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
