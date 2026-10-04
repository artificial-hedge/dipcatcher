import pytest


@pytest.mark.parametrize(
    "name",
    [
        "jiangshi_qa_studies",
        "huli_qa_studies",
        "mogwai_qa_studies",
        "yaoguai_qa_studies",
        "dijiang_qa_studies",
        "zhuyin_qa_studies",
    ],
)
def test_w1695_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
