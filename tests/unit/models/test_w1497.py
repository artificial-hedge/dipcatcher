import pytest


@pytest.mark.parametrize(
    "name",
    [
        "numbat2_qa_studies",
        "bettong_qa_studies",
        "potoroo_qa_studies",
        "woylie_qa_studies",
        "cuscus_qa_studies",
        "pademelon_qa_studies",
    ],
)
def test_w1497_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
