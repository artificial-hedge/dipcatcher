import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hooded_seal_qa_studies",
        "ribbon_seal_qa_studies",
        "ross_seal_qa_studies",
        "crabeater_qa_studies",
        "bearded_seal_qa_studies",
        "ringed_seal_qa_studies",
    ],
)
def test_w1588_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
