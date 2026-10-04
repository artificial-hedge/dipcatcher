import pytest


@pytest.mark.parametrize(
    "name",
    [
        "conger_qa_studies",
        "garden_eel_qa_studies",
        "hagfish_qa_studies",
        "moray_qa_studies",
        "ribbon_eel_qa_studies",
        "lamprey_qa_studies",
    ],
)
def test_w1568_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
