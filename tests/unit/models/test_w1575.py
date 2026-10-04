import pytest


@pytest.mark.parametrize(
    "name",
    [
        "vole_qa_studies",
        "lemming_qa_studies",
        "degu_qa_studies",
        "chinchilla_qa_studies",
        "gerbil_qa_studies",
        "hamster_qa_studies",
    ],
)
def test_w1575_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
