import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sedna_qa_studies",
        "torngarsuk_qa_studies",
        "agloolik_qa_studies",
        "aumanil_qa_studies",
        "nuktessien_qa_studies",
        "tekkeitsertok_qa_studies",
    ],
)
def test_w1744_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
