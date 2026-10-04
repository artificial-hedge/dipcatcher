import pytest


@pytest.mark.parametrize(
    "name",
    [
        "earthworm_qa_studies",
        "feather_duster_qa_studies",
        "leech_qa_studies",
        "polychaete_qa_studies",
        "lugworm_qa_studies",
        "ragworm_qa_studies",
    ],
)
def test_w1565_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
