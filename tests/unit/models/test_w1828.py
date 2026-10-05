import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ilmatar2_qa_studies",
        "jumala2_qa_studies",
        "metsanhiisi2_qa_studies",
        "peikko2_qa_studies",
        "ajatar2_qa_studies",
        "otso2_qa_studies",
    ],
)
def test_w1828_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
