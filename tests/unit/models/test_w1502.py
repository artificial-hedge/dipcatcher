import pytest


@pytest.mark.parametrize(
    "name",
    [
        "anteater_qa_studies",
        "coatimundi_qa_studies",
        "opossum_qa_studies",
        "kinkajou_qa_studies",
        "tamandua_qa_studies",
        "paca_qa_studies",
    ],
)
def test_w1502_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
