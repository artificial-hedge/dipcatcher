import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gadfly_qa_studies",
        "diving_petrel_qa_studies",
        "manx_qa_studies",
        "anhinga_qa_studies",
        "darter_qa_studies",
        "mollymawk_qa_studies",
    ],
)
def test_w1541_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
