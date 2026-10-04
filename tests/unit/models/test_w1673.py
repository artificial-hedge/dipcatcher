import pytest


@pytest.mark.parametrize(
    "name",
    [
        "rakshasa_qa_studies",
        "yaksha_qa_studies",
        "gandharva_qa_studies",
        "apsara_qa_studies",
        "asura_qa_studies",
        "naga_qa_studies",
    ],
)
def test_w1673_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
