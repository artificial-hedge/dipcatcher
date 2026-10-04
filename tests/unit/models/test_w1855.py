import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ataecina_qa_studies",
        "endovellicus_qa_studies",
        "trebaruna_qa_studies",
        "nabia_qa_studies",
        "bandua_qa_studies",
        "cariocecus_qa_studies",
    ],
)
def test_w1855_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
