import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sci_q_studies",
        "arc_challenge_studies",
        "openbook_qa_studies",
        "med_qa_studies",
        "pubmed_qa_studies",
        "bio_qa_studies",
    ],
)
def test_w1340_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
