import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ghost_mantis_qa_studies",
        "shield_mantis_qa_studies",
        "orchid_mantis_qa_studies",
        "empusa_qa_studies",
        "mantidfly_qa_studies",
        "praying_mantis_qa_studies",
    ],
)
def test_w1519_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
