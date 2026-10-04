import pytest


@pytest.mark.parametrize(
    "name",
    [
        "potto_qa_studies",
        "galago_qa_studies",
        "indri_qa_studies",
        "loris_qa_studies",
        "bushbaby_qa_studies",
        "tarsier_qa_studies",
    ],
)
def test_w1592_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
