import pytest


@pytest.mark.parametrize(
    "name",
    [
        "anansi_qa_studies",
        "mamiwata_qa_studies",
        "sasabonsam_qa_studies",
        "tokoloshe_qa_studies",
        "kalulu_qa_studies",
        "impundulu_qa_studies",
    ],
)
def test_w1681_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
