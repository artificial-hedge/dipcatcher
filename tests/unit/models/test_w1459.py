import pytest


@pytest.mark.parametrize(
    "name",
    [
        "caribou_qa_studies",
        "penguin_qa_studies",
        "musk_ox_qa_studies",
        "polar_bear_qa_studies",
        "reindeer_qa_studies",
        "arctic_fox_qa_studies",
    ],
)
def test_w1459_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
