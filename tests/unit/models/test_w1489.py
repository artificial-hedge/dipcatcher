import pytest


@pytest.mark.parametrize(
    "name",
    [
        "heather_qa_studies",
        "lilac_qa_studies",
        "lavender_qa_studies",
        "marigold_qa_studies",
        "primrose_qa_studies",
        "clover_qa_studies",
    ],
)
def test_w1489_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
