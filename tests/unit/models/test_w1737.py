import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tlaloc_qa_studies",
        "huitzilopochtli_qa_studies",
        "coatlicue_qa_studies",
        "mictlantecuhtli_qa_studies",
        "tonatiuh_qa_studies",
        "coyolxauhqui_qa_studies",
    ],
)
def test_w1737_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
