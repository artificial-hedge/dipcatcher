import pytest


@pytest.mark.parametrize(
    "name",
    [
        "amenokal_qa_studies",
        "imajeghen_qa_studies",
        "ammonion_qa_studies",
        "atlas_deity_qa_studies",
        "tritogeneia_qa_studies",
        "melqart_libya_qa_studies",
    ],
)
def test_w1885_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
