import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nereus_qa_studies",
        "proteus_qa_studies",
        "triton_qa_studies",
        "pontus_qa_studies",
        "thaumas_qa_studies",
        "phorcys_qa_studies",
    ],
)
def test_w1752_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
