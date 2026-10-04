import pytest


@pytest.mark.parametrize(
    "name",
    [
        "eagle_qa_studies",
        "owl_qa_studies",
        "falcon_qa_studies",
        "raven_qa_studies",
        "swan_qa_studies",
        "crane_qa_studies",
    ],
)
def test_w1441_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
