import pytest


@pytest.mark.parametrize(
    "name",
    [
        "furcas_qa_studies",
        "alloces_qa_studies",
        "balam_qa_studies",
        "camio_qa_studies",
        "gaap_qa_studies",
        "foras_qa_studies",
    ],
)
def test_w1952_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
