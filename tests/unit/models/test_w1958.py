import pytest


@pytest.mark.parametrize(
    "name",
    [
        "marbas_qa_studies",
        "buer_qa_studies",
        "bathin_qa_studies",
        "sallos_qa_studies",
        "purson_qa_studies",
        "marax_qa_studies",
    ],
)
def test_w1958_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
