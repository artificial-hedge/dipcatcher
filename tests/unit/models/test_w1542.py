import pytest


@pytest.mark.parametrize(
    "name",
    [
        "seriema_qa_studies",
        "accipiter_qa_studies",
        "harpy_qa_studies",
        "lammergeier_qa_studies",
        "bateleur_qa_studies",
        "falconet_qa_studies",
    ],
)
def test_w1542_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
