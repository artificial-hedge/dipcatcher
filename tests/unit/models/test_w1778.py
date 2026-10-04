import pytest


@pytest.mark.parametrize(
    "name",
    [
        "vishnu_qa_studies",
        "rama_qa_studies",
        "hanuman_qa_studies",
        "sita_qa_studies",
        "lakshmi_qa_studies",
        "parvati_qa_studies",
    ],
)
def test_w1778_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
