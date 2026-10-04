import pytest


@pytest.mark.parametrize(
    "name",
    [
        "beacon_qa_studies",
        "monolith_qa_studies",
        "blizzard_qa_studies",
        "spire_qa_studies",
        "tempest_qa_studies",
        "abyss_qa_studies",
    ],
)
def test_w1464_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
