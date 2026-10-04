import pytest


@pytest.mark.parametrize(
    "name",
    [
        "opal_qa_studies",
        "tourmaline_qa_studies",
        "garnet_qa_studies",
        "aquamarine_qa_studies",
        "tanzanite_qa_studies",
        "ruby_qa_studies",
    ],
)
def test_w1529_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
