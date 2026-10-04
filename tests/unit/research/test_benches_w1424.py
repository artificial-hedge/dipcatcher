import pytest

from quant_fund.research import benches_w1424


@pytest.mark.parametrize(
    "fam",
    [
        "bench_blueprint_qa_studies_family",
        "bench_design_qa_studies_family",
        "bench_format_qa_studies_family",
        "bench_layout_qa_studies_family",
        "bench_pattern_qa_studies_family",
        "bench_schema_qa_studies_family",
    ],
)
def test_benches_w1424(fam):
    out = getattr(benches_w1424, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
