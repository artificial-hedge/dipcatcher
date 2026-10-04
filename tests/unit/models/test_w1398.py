import pytest


@pytest.mark.parametrize(
    "name",
    [
        "finqa_lite_studies",
        "infotabs_studies",
        "hybridqa_lite_studies",
        "ottqa_lite_studies",
        "tab_cwq_studies",
        "doc2dial_studies",
    ],
)
def test_w1398_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
