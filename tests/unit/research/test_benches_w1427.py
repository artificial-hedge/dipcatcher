import pytest

from quant_fund.research import benches_w1427


@pytest.mark.parametrize(
    "fam",
    [
        "bench_canyon_qa_studies_family",
        "bench_coast_qa_studies_family",
        "bench_desert_qa_studies_family",
        "bench_field_qa_studies_family",
        "bench_forest_qa_studies_family",
        "bench_glacier_qa_studies_family",
    ],
)
def test_benches_w1427(fam):
    out = getattr(benches_w1427, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
