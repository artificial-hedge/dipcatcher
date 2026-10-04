import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hecate_qa_studies",
        "hypnos_qa_studies",
        "thanatos_qa_studies",
        "zephyrus_qa_studies",
        "helios_qa_studies",
        "selene_qa_studies",
    ],
)
def test_w1772_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
