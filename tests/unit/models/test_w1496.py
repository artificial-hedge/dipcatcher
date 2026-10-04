import pytest


@pytest.mark.parametrize(
    "name",
    [
        "moorhen_qa_studies",
        "snipe_qa_studies",
        "railbird_qa_studies",
        "turnstone_qa_studies",
        "lapwing_qa_studies",
        "jacana_qa_studies",
    ],
)
def test_w1496_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
