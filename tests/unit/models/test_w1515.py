import pytest


@pytest.mark.parametrize(
    "name",
    [
        "rhino_beetle_qa_studies",
        "dung_beetle_qa_studies",
        "click_beetle_qa_studies",
        "ground_beetle_qa_studies",
        "tiger_beetle_qa_studies",
        "stag_beetle_qa_studies",
    ],
)
def test_w1515_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
