import pytest

from quant_fund.research import benches_w1752


@pytest.mark.parametrize(
    "fam",
    [
        "bench_nereus_qa_studies_family",
        "bench_phorcys_qa_studies_family",
        "bench_pontus_qa_studies_family",
        "bench_proteus_qa_studies_family",
        "bench_thaumas_qa_studies_family",
        "bench_triton_qa_studies_family",
    ],
)
def test_benches_w1752(fam):
    out = getattr(benches_w1752, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
