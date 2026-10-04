import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bannik_qa_studies",
        "dvorovoi_qa_studies",
        "ovinnik_qa_studies",
        "vila_qa_studies",
        "mora_qa_studies",
        "poludnica_qa_studies",
    ],
)
def test_w1678_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
