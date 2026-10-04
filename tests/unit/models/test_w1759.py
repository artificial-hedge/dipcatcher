import pytest


@pytest.mark.parametrize(
    "name",
    [
        "xochipilli_qa_studies",
        "tonaca_qa_studies",
        "metzli_qa_studies",
        "citlali_qa_studies",
        "tepoz_qa_studies",
        "malinal_qa_studies",
    ],
)
def test_w1759_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
