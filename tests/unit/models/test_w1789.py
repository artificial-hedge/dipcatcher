import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hermes_qa_studies",
        "hebe_qa_studies",
        "ganymede_qa_studies",
        "momus_qa_studies",
        "oneiros_qa_studies",
        "eris_qa_studies",
    ],
)
def test_w1789_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
