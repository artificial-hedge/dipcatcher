import pytest


@pytest.mark.parametrize(
    "name",
    [
        "course_qa_studies",
        "homework_qa_studies",
        "exam_qa_studies",
        "lecture_qa_studies",
        "seminar_qa_studies",
        "class_qa_studies",
    ],
)
def test_w1423_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
