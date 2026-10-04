import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mackerel_qa_studies",
        "anchovy_qa_studies",
        "bonito_qa_studies",
        "kingfish_qa_studies",
        "sardine_qa_studies",
        "herring_qa_studies",
    ],
)
def test_w1559_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
