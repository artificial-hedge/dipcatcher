import pytest

from quant_fund.research import benches_w1727


@pytest.mark.parametrize(
    "fam",
    [
        "bench_erlug_qa_studies_family",
        "bench_etseg_qa_studies_family",
        "bench_manzan_qa_studies_family",
        "bench_otgon_qa_studies_family",
        "bench_tenger_qa_studies_family",
        "bench_ulgan_qa_studies_family",
    ],
)
def test_benches_w1727(fam):
    out = getattr(benches_w1727, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
