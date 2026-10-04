import pytest

from quant_fund.research import benches_w1817


@pytest.mark.parametrize(
    "fam",
    [
        "bench_durga2_qa_studies_family",
        "bench_ganga2_qa_studies_family",
        "bench_kali2_qa_studies_family",
        "bench_lakshmi2_qa_studies_family",
        "bench_parvati2_qa_studies_family",
        "bench_saraswati2_qa_studies_family",
    ],
)
def test_benches_w1817(fam):
    out = getattr(benches_w1817, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
