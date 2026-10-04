import pytest


@pytest.mark.parametrize(
    "name",
    [
        "lesser_kudu_qa_studies",
        "mountain_nyala_qa_studies",
        "sitatunga_qa_studies",
        "greater_kudu_qa_studies",
        "bontebok_qa_studies",
        "bushbuck_qa_studies",
    ],
)
def test_w1598_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
