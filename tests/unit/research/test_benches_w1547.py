import pytest

from quant_fund.research import benches_w1547


@pytest.mark.parametrize(
    "fam",
    [
        "bench_crowned_pigeon_qa_studies_family",
        "bench_cuckoo_dove_qa_studies_family",
        "bench_emerald_dove_qa_studies_family",
        "bench_fruit_dove_qa_studies_family",
        "bench_ground_dove_qa_studies_family",
        "bench_quail_dove_qa_studies_family",
    ],
)
def test_benches_w1547(fam):
    out = getattr(benches_w1547, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
