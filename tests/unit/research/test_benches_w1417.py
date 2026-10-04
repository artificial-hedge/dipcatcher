import pytest

from quant_fund.research import benches_w1417


@pytest.mark.parametrize(
    "fam",
    [
        "bench_checklist_qa_studies_family",
        "bench_flow_qa_studies_family",
        "bench_guide_qa_studies_family",
        "bench_howto_qa_studies_family",
        "bench_instruct_qa_studies_family",
        "bench_lesson_qa_studies_family",
    ],
)
def test_benches_w1417(fam):
    out = getattr(benches_w1417, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
