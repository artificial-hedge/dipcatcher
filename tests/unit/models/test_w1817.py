import pytest


@pytest.mark.parametrize(
    "name",
    [
        "parvati2_qa_studies",
        "lakshmi2_qa_studies",
        "saraswati2_qa_studies",
        "durga2_qa_studies",
        "kali2_qa_studies",
        "ganga2_qa_studies",
    ],
)
def test_w1817_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
