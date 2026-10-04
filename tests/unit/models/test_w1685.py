import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ettin_qa_studies",
        "nokken_qa_studies",
        "vette_qa_studies",
        "fylgja_qa_studies",
        "landvaettir_qa_studies",
        "seidr_qa_studies",
    ],
)
def test_w1685_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
