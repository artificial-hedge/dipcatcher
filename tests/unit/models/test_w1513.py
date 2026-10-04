import pytest


@pytest.mark.parametrize(
    "name",
    [
        "screech_owl_qa_studies",
        "barred_owl_qa_studies",
        "snowy_owl_qa_studies",
        "eagle_owl_qa_studies",
        "tawny_owl_qa_studies",
        "barnowl_qa_studies",
    ],
)
def test_w1513_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
