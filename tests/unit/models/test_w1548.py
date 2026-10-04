import pytest


@pytest.mark.parametrize(
    "name",
    [
        "guira_qa_studies",
        "hoatzin_qa_studies",
        "ani_qa_studies",
        "turaco_qa_studies",
        "coua_qa_studies",
        "malkoha_qa_studies",
    ],
)
def test_w1548_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
