import pytest

from quant_fund.research import benches_w1800


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aeneas2_qa_studies_family",
        "bench_evander2_qa_studies_family",
        "bench_lavinia2_qa_studies_family",
        "bench_remus2_qa_studies_family",
        "bench_romulus2_qa_studies_family",
        "bench_turnus2_qa_studies_family",
    ],
)
def test_benches_w1800(fam):
    out = getattr(benches_w1800, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
