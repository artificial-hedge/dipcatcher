import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dragonfly_qa_studies",
        "ladybug_qa_studies",
        "grasshopper_qa_studies",
        "mantis_qa_studies",
        "scorpion_qa_studies",
        "cicada_qa_studies",
    ],
)
def test_w1475_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
