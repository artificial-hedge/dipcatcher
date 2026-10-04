import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nuwa_qa_studies",
        "fuxi_qa_studies",
        "shennong_qa_studies",
        "yandi_qa_studies",
        "huangdi_qa_studies",
        "xihe_qa_studies",
    ],
)
def test_w1779_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
