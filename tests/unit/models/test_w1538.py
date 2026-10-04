import pytest


@pytest.mark.parametrize(
    "name",
    [
        "limpkin_qa_studies",
        "crowned_crane_qa_studies",
        "whooping_qa_studies",
        "finfoot_qa_studies",
        "demoiselle_qa_studies",
        "trumpeter_qa_studies",
    ],
)
def test_w1538_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
