import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gylfaginning_prose_qa_studies",
        "edda_lore_qa_studies",
        "rigsthula_qa_studies",
        "hyndluljod_qa_studies",
        "alvissmal_qa_studies",
        "haddingjar_qa_studies",
    ],
)
def test_w1889_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
