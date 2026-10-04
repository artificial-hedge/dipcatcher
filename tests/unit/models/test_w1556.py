import pytest


@pytest.mark.parametrize(
    "name",
    [
        "velvet_worm_qa_studies",
        "springtail_qa_studies",
        "pillbug_qa_studies",
        "bristletail_qa_studies",
        "silverfish_qa_studies",
        "woodlouse_qa_studies",
    ],
)
def test_w1556_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
