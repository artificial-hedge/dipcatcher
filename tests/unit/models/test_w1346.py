import pytest


@pytest.mark.parametrize(
    "name",
    [
        "lc_quad_studies",
        "mintaka_qa_studies",
        "kqa_pro_studies",
        "graph_questions_studies",
        "grail_qa_studies",
        "spinach_qa_studies",
    ],
)
def test_w1346_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
