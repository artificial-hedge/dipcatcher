import pytest

from quant_fund.research import benches_w1831


@pytest.mark.parametrize(
    "fam",
    [
        "bench_asherah3_qa_studies_family",
        "bench_baal3_qa_studies_family",
        "bench_el3_qa_studies_family",
        "bench_kothar3_qa_studies_family",
        "bench_lotan3_qa_studies_family",
        "bench_mot3_qa_studies_family",
    ],
)
def test_benches_w1831(fam):
    out = getattr(benches_w1831, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
