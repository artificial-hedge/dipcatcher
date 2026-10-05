import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bogdan2_qa_studies",
        "radegast2_qa_studies",
        "ziva2_qa_studies",
        "kupalo2_qa_studies",
        "rod2_qa_studies",
        "bereginia2_qa_studies",
    ],
)
def test_w1805_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
