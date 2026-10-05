import pytest


@pytest.mark.parametrize(
    "name",
    [
        "amdusias_qa_studies",
        "decarabia_qa_studies",
        "seere_qa_studies",
        "dantalion_qa_studies",
        "andromalius_qa_studies",
        "malphas_qa_studies",
    ],
)
def test_w1955_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
