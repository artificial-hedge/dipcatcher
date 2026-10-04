import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sphinx_2_qa_studies",
        "manticore_2_qa_studies",
        "chimera_2_qa_studies",
        "basilisk_2_qa_studies",
        "cockatrice_qa_studies",
        "wyvern_2_qa_studies",
    ],
)
def test_w1642_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
