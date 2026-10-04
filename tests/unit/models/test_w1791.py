import pytest


@pytest.mark.parametrize(
    "name",
    [
        "frey_qa_studies",
        "freya2_qa_studies",
        "njord_qa_studies",
        "tyr2_qa_studies",
        "magni_qa_studies",
        "modi_qa_studies",
    ],
)
def test_w1791_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
