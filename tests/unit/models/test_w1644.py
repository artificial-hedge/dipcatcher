import pytest


@pytest.mark.parametrize(
    "name",
    [
        "medusa_2_qa_studies",
        "cyclops_2_qa_studies",
        "hydra_3_qa_studies",
        "scylla_qa_studies",
        "charybdis_qa_studies",
        "siren_2_qa_studies",
    ],
)
def test_w1644_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
