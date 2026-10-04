import pytest


@pytest.mark.parametrize(
    "name",
    [
        "atlas_qa_studies",
        "jeopardy_qa_studies",
        "idiom_qa_studies",
        "misc_qa_studies",
        "myth_qa_studies",
        "almanac_qa_studies",
    ],
)
def test_w1418_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
