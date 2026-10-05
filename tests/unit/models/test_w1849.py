import pytest


@pytest.mark.parametrize(
    "name",
    [
        "aster2_qa_studies",
        "mahrem_qa_studies",
        "beher_qa_studies",
        "medr_qa_studies",
        "alouq_qa_studies",
        "ruda_qa_studies",
    ],
)
def test_w1849_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
