import pytest


@pytest.mark.parametrize(
    "name",
    [
        "houyi_qa_studies",
        "yutu_qa_studies",
        "wenchang_qa_studies",
        "guandi_qa_studies",
        "zao_qa_studies",
        "aoqin_qa_studies",
    ],
)
def test_w1697_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
