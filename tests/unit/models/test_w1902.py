import pytest


@pytest.mark.parametrize(
    "name",
    [
        "shuten_doji_qa_studies",
        "ao_andon_qa_studies",
        "hannya_oni_qa_studies",
        "tsuchigumo_qa_studies",
        "nurarihyon_qa_studies",
        "kamaitachi_qa_studies",
    ],
)
def test_w1902_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
