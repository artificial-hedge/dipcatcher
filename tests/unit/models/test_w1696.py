import pytest


@pytest.mark.parametrize(
    "name",
    [
        "xiwangmu_qa_studies",
        "dongwanggong_qa_studies",
        "nuwa_qa_studies",
        "fuxi_qa_studies",
        "shennong_qa_studies",
        "kuafu_qa_studies",
    ],
)
def test_w1696_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
