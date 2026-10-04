import pytest

from quant_fund.research import benches_w1810


@pytest.mark.parametrize(
    "fam",
    [
        "bench_elegba2_qa_studies_family",
        "bench_obatala2_qa_studies_family",
        "bench_orunmila2_qa_studies_family",
        "bench_osun2_qa_studies_family",
        "bench_oya2_qa_studies_family",
        "bench_shango2_qa_studies_family",
    ],
)
def test_benches_w1810(fam):
    out = getattr(benches_w1810, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
