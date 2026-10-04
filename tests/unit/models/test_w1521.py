import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dendrobium_qa_studies",
        "cymbidium_qa_studies",
        "phalaenopsis_qa_studies",
        "paphiopedilum_qa_studies",
        "oncidium_qa_studies",
        "cattleya_qa_studies",
    ],
)
def test_w1521_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
