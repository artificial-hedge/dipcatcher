import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nian_qa_studies",
        "baigujing_qa_studies",
        "xiangliu_qa_studies",
        "jiuying_qa_studies",
        "wuzhiqi_qa_studies",
        "hanba_qa_studies",
    ],
)
def test_w1923_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
