import pytest


@pytest.mark.parametrize(
    "name",
    [
        "alp_qa_studies",
        "mahr_qa_studies",
        "doppelganger_qa_studies",
        "poltergeist_qa_studies",
        "kobold_qa_studies",
        "tatzelwurm_qa_studies",
    ],
)
def test_w1916_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
