import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sabrewing_qa_studies",
        "fairy_qa_studies",
        "lancebill_qa_studies",
        "coquette_qa_studies",
        "jacobin_qa_studies",
        "sheartail_qa_studies",
    ],
)
def test_w1518_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
