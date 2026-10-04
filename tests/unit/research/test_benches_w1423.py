import pytest

from quant_fund.research import benches_w1423


@pytest.mark.parametrize(
    "fam",
    [
        "bench_class_qa_studies_family",
        "bench_course_qa_studies_family",
        "bench_exam_qa_studies_family",
        "bench_homework_qa_studies_family",
        "bench_lecture_qa_studies_family",
        "bench_seminar_qa_studies_family",
    ],
)
def test_benches_w1423(fam):
    out = getattr(benches_w1423, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
