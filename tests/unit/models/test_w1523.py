import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fescue_qa_studies",
        "bluegrass_qa_studies",
        "ryegrass_qa_studies",
        "switchgrass_qa_studies",
        "miscanthus_qa_studies",
        "pampas_qa_studies",
    ],
)
def test_w1523_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
