import pytest


@pytest.mark.parametrize(
    "name",
    [
        "zamolxis_qa_studies",
        "kezion_qa_studies",
        "bendis_qa_studies",
        "derzelas_qa_studies",
        "sabazios_qa_studies",
        "darzalas_qa_studies",
    ],
)
def test_w1715_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
