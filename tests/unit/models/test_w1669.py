import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ghouling_qa_studies",
        "ikugan_qa_studies",
        "kataw_qa_studies",
        "lambana_qa_studies",
        "sarimanok_qa_studies",
        "tamahaling_qa_studies",
    ],
)
def test_w1669_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
