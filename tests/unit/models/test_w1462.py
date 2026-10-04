import pytest


@pytest.mark.parametrize(
    "name",
    [
        "jellyfish_qa_studies",
        "seahorse_qa_studies",
        "octopus_qa_studies",
        "squid_qa_studies",
        "stingray_qa_studies",
        "crab_qa_studies",
    ],
)
def test_w1462_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
