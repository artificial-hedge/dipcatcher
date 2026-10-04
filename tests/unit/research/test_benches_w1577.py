import pytest

from quant_fund.research import benches_w1577


@pytest.mark.parametrize(
    "fam",
    [
        "bench_civet_qa_studies_family",
        "bench_genet_qa_studies_family",
        "bench_manul_qa_studies_family",
        "bench_mongoose_qa_studies_family",
        "bench_sloth_bear_qa_studies_family",
        "bench_suricate_qa_studies_family",
    ],
)
def test_benches_w1577(fam):
    out = getattr(benches_w1577, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
