import pytest


@pytest.mark.parametrize(
    "name",
    [
        "curupira_qa_studies",
        "saci_qa_studies",
        "boitata_qa_studies",
        "iara_qa_studies",
        "boto_qa_studies",
        "mapinguari_qa_studies",
    ],
)
def test_w1901_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
