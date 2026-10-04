import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ullr_qa_studies",
        "sif_qa_studies",
        "idun_qa_studies",
        "bragi_qa_studies",
        "forseti_qa_studies",
        "vidar_qa_studies",
    ],
)
def test_w1747_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
