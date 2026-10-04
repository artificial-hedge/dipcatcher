import pytest


@pytest.mark.parametrize(
    "name",
    [
        "counter_qa_studies",
        "everyday_qa_studies",
        "custom_qa_studies",
        "folk_qa_studies",
        "moral_qa_studies",
        "afford_qa_studies",
    ],
)
def test_w1414_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
