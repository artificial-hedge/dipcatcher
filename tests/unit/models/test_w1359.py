import pytest


@pytest.mark.parametrize(
    "name",
    [
        "babi_lite_studies",
        "web_questions_studies",
        "curious_qa_studies",
        "qasper_lite_studies",
        "scifact_lite_studies",
        "argu_ana_studies",
    ],
)
def test_w1359_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
