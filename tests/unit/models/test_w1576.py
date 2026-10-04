import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fossa_qa_studies",
        "kusimanse_qa_studies",
        "sun_bear_qa_studies",
        "binturong_qa_studies",
        "honey_badger_qa_studies",
        "maned_wolf_qa_studies",
    ],
)
def test_w1576_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
