import pytest


@pytest.mark.parametrize(
    "name",
    [
        "epona2_qa_studies",
        "maponos2_qa_studies",
        "andrasta2_qa_studies",
        "rosmerta2_qa_studies",
        "etercuni2_qa_studies",
        "borvo2_qa_studies",
    ],
)
def test_w1803_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
