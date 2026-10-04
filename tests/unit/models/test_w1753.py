import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gonggong_qa_studies",
        "zhurong_qa_studies",
        "changxi_qa_studies",
        "xihe_qa_studies",
        "yinglong_qa_studies",
        "chiyou_qa_studies",
    ],
)
def test_w1753_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
