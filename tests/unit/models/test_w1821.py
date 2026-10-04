import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ebisu2_qa_studies",
        "daikoku2_qa_studies",
        "benzaiten2_qa_studies",
        "hotei2_qa_studies",
        "juroujin2_qa_studies",
        "fukurokuju2_qa_studies",
    ],
)
def test_w1821_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
