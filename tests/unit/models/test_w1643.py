import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gryphon_qa_studies",
        "hippogryph_qa_studies",
        "centaur_2_qa_studies",
        "minotaur_2_qa_studies",
        "satyr_2_qa_studies",
        "harpy_2_qa_studies",
    ],
)
def test_w1643_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
