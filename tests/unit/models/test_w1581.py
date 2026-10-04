import pytest


@pytest.mark.parametrize(
    "name",
    [
        "elephant_seal_qa_studies",
        "leopard_seal_qa_studies",
        "weddell_qa_studies",
        "fur_seal_qa_studies",
        "monk_seal_qa_studies",
        "harp_seal_qa_studies",
    ],
)
def test_w1581_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
