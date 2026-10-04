import pytest


@pytest.mark.parametrize(
    "name",
    [
        "shango2_qa_studies",
        "oya2_qa_studies",
        "osun2_qa_studies",
        "obatala2_qa_studies",
        "elegba2_qa_studies",
        "orunmila2_qa_studies",
    ],
)
def test_w1810_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
