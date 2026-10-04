import pytest


@pytest.mark.parametrize(
    "name",
    [
        "copperhead_qa_studies",
        "cottonmouth_qa_studies",
        "bushmaster_qa_studies",
        "rattlesnake_qa_studies",
        "coral_snake_qa_studies",
        "fer_de_lance_qa_studies",
    ],
)
def test_w1551_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
