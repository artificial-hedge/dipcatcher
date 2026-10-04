import pytest

from quant_fund.research import benches_w1414


@pytest.mark.parametrize(
    "fam",
    [
        "bench_afford_qa_studies_family",
        "bench_counter_qa_studies_family",
        "bench_custom_qa_studies_family",
        "bench_everyday_qa_studies_family",
        "bench_folk_qa_studies_family",
        "bench_moral_qa_studies_family",
    ],
)
def test_benches_w1414(fam):
    out = getattr(benches_w1414, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
