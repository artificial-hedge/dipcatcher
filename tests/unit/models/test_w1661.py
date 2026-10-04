import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fenrir_qa_studies",
        "gullinbursti_qa_studies",
        "huginn_qa_studies",
        "muninn_qa_studies",
        "hraesvelgr_qa_studies",
        "draugr_qa_studies",
    ],
)
def test_w1661_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
