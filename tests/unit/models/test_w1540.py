import pytest


@pytest.mark.parametrize(
    "name",
    [
        "great_egret_qa_studies",
        "glossy_ibis_qa_studies",
        "squacco_qa_studies",
        "sacred_ibis_qa_studies",
        "cattle_egret_qa_studies",
        "snowy_egret_qa_studies",
    ],
)
def test_w1540_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
