import pytest


@pytest.mark.parametrize(
    "name",
    [
        "betobeto_qa_studies",
        "hyakume_qa_studies",
        "waira_qa_studies",
        "buruburu_qa_studies",
        "uwan_qa_studies",
        "shachihoko_qa_studies",
    ],
)
def test_w1905_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
