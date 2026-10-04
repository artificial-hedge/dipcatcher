import pytest

from quant_fund.research import benches_w1887


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dakini_qa_studies_family",
        "bench_indra_hindu_qa_studies_family",
        "bench_jamshid_qa_studies_family",
        "bench_varuna_qa_studies_family",
        "bench_vritra_qa_studies_family",
        "bench_zurvan_qa_studies_family",
    ],
)
def test_benches_w1887(fam):
    out = getattr(benches_w1887, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
