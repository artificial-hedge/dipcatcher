import pytest


@pytest.mark.parametrize(
    "name",
    [
        "knifefish_qa_studies",
        "arapaima_qa_studies",
        "pacu_qa_studies",
        "electric_eel_qa_studies",
        "tetra_qa_studies",
        "oscar_qa_studies",
    ],
)
def test_w1557_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
