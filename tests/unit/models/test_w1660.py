import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hydra_qa_studies",
        "sphinx_qa_studies",
        "griffin_qa_studies",
        "centaur_qa_studies",
        "cyclops_qa_studies",
        "medusa_qa_studies",
    ],
)
def test_w1660_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
