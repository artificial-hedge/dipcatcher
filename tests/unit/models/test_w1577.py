import pytest


@pytest.mark.parametrize(
    "name",
    [
        "genet_qa_studies",
        "mongoose_qa_studies",
        "sloth_bear_qa_studies",
        "civet_qa_studies",
        "suricate_qa_studies",
        "manul_qa_studies",
    ],
)
def test_w1577_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
