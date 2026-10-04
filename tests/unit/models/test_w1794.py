import pytest


@pytest.mark.parametrize(
    "name",
    [
        "rongomai_qa_studies",
        "uenuku2_qa_studies",
        "maru2_qa_studies",
        "wairere_qa_studies",
        "pere_qa_studies",
        "tuhi2_qa_studies",
    ],
)
def test_w1794_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
