import pytest

from quant_fund.research import benches_w1534


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cuckoo_qa_studies_family",
        "bench_frogmouth_qa_studies_family",
        "bench_koel_qa_studies_family",
        "bench_nighthawk_qa_studies_family",
        "bench_nightjar_qa_studies_family",
        "bench_roadrunner_qa_studies_family",
    ],
)
def test_benches_w1534(fam):
    out = getattr(benches_w1534, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
