import pytest


@pytest.mark.parametrize(
    "name",
    [
        "marbled_cat_qa_studies",
        "bay_cat_qa_studies",
        "andean_cat_qa_studies",
        "flat_headed_qa_studies",
        "pampas_cat_qa_studies",
        "geoffroys_qa_studies",
    ],
)
def test_w1595_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
