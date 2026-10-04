import pytest


@pytest.mark.parametrize(
    "name",
    [
        "catfish_qa_studies",
        "piranha_qa_studies",
        "cod_qa_studies",
        "salmon_qa_studies",
        "tuna_qa_studies",
        "barracuda_qa_studies",
    ],
)
def test_w1456_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
