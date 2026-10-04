import pytest


@pytest.mark.parametrize(
    "name",
    [
        "haworthia_qa_studies",
        "aloe_qa_studies",
        "lithops_qa_studies",
        "agave_qa_studies",
        "sedum_qa_studies",
        "echeveria_qa_studies",
    ],
)
def test_w1522_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
