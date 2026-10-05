import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sulis_qa_studies",
        "cocidius_qa_studies",
        "nemetona_qa_studies",
        "rigisamus_qa_studies",
        "belatucadrus_qa_studies",
        "maponus_qa_studies",
    ],
)
def test_w1857_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
