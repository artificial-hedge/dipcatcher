import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gwishin_qa_studies",
        "dokkaebi_qa_studies",
        "gumiho_qa_studies",
        "mul_gwishin_qa_studies",
        "cheonyeo_gwishin_qa_studies",
        "oeggwi_qa_studies",
    ],
)
def test_w1944_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
