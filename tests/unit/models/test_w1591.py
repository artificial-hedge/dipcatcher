import pytest


@pytest.mark.parametrize(
    "name",
    [
        "titi_qa_studies",
        "uakari_qa_studies",
        "squirrel_monkey_qa_studies",
        "capuchin_qa_studies",
        "saki_qa_studies",
        "woolly_qa_studies",
    ],
)
def test_w1591_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
