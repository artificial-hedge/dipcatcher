import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cheetah_qa_studies",
        "leopard_qa_studies",
        "fox_qa_studies",
        "lion_qa_studies",
        "wolf_qa_studies",
        "bear_qa_studies",
    ],
)
def test_w1447_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
