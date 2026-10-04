import pytest


@pytest.mark.parametrize(
    "name",
    [
        "contest_qa_studies",
        "hobby_qa_studies",
        "game_qa_studies",
        "leisure_qa_studies",
        "match_qa_studies",
        "challenge_qa_studies",
    ],
)
def test_w1426_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
