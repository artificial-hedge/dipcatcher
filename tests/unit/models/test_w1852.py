import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gurzil_qa_studies",
        "macurgum_qa_studies",
        "aulisua_qa_studies",
        "lallus_qa_studies",
        "iguc_qa_studies",
        "melyakina_qa_studies",
    ],
)
def test_w1852_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
