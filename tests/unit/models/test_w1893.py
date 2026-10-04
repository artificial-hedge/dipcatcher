import pytest


@pytest.mark.parametrize(
    "name",
    [
        "galla_demon_qa_studies",
        "rabisu_qa_studies",
        "alu_demon_qa_studies",
        "lilitu_qa_studies",
        "ardat_lili_qa_studies",
        "lamashtu_qa_studies",
    ],
)
def test_w1893_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
