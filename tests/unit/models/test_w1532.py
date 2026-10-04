import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bee_eater_qa_studies",
        "motmot_qa_studies",
        "roller_qa_studies",
        "tody_qa_studies",
        "jacamar_qa_studies",
        "kookaburra_qa_studies",
    ],
)
def test_w1532_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
