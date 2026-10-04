import pytest


@pytest.mark.parametrize(
    "name",
    [
        "chamois_qa_studies",
        "tahr_qa_studies",
        "bharal_qa_studies",
        "ibex_qa_studies",
        "goral_qa_studies",
        "serow_qa_studies",
    ],
)
def test_w1597_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
