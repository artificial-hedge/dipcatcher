import pytest


@pytest.mark.parametrize(
    "name",
    [
        "storm_petrel_qa_studies",
        "shag_qa_studies",
        "noddie_qa_studies",
        "skimmer_qa_studies",
        "murre_qa_studies",
        "prion_qa_studies",
    ],
)
def test_w1504_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
