import pytest


@pytest.mark.parametrize(
    "name",
    [
        "leraje_qa_studies",
        "beleth_qa_studies",
        "botis_qa_studies",
        "sitri_qa_studies",
        "eligos_qa_studies",
        "zepar_qa_studies",
    ],
)
def test_w1950_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
