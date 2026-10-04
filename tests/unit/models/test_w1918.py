import pytest


@pytest.mark.parametrize(
    "name",
    [
        "redcap_qa_studies",
        "dunter_qa_studies",
        "bocan_qa_studies",
        "fuath_qa_studies",
        "seonaidh_qa_studies",
        "wraith_qa_studies",
    ],
)
def test_w1918_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
