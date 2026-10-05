import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kilmoulis_qa_studies",
        "shellycoat_qa_studies",
        "geancanach_qa_studies",
        "fachan_qa_studies",
        "caointeach_qa_studies",
        "bodach_qa_studies",
    ],
)
def test_w1919_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
