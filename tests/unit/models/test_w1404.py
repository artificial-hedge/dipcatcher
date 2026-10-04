import pytest


@pytest.mark.parametrize(
    "name",
    [
        "how2qa_lite_studies",
        "msrvtt_qa_studies",
        "movie_qa_lite_studies",
        "nextqa_lite_studies",
        "star_qa_lite_studies",
        "activitynet_qa_studies",
    ],
)
def test_w1404_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
