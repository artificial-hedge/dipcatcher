import pytest


@pytest.mark.parametrize(
    "name",
    [
        "langur_qa_studies",
        "marmoset_qa_studies",
        "macaque_qa_studies",
        "gibbon_qa_studies",
        "tamarin_qa_studies",
        "lemur_qa_studies",
    ],
)
def test_w1574_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
