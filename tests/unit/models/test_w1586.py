import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tsessebe_qa_studies",
        "reedbuck_qa_studies",
        "beira_qa_studies",
        "gemsbok_qa_studies",
        "oribi_qa_studies",
        "madoqua_qa_studies",
    ],
)
def test_w1586_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
