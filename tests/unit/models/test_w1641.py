import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fenghuang_qa_studies",
        "hundun_qa_studies",
        "taotie_qa_studies",
        "taowu_qa_studies",
        "qiongqi_qa_studies",
        "bixie_qa_studies",
    ],
)
def test_w1641_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
