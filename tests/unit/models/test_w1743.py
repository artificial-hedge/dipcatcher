import pytest


@pytest.mark.parametrize(
    "name",
    [
        "unkulunkulu_qa_studies",
        "mamlambo_qa_studies",
        "tikoloshe_qa_studies",
        "impundulu_qa_studies",
        "inkanyamba_qa_studies",
        "usilosimapundu_qa_studies",
    ],
)
def test_w1743_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
