import pytest


@pytest.mark.parametrize(
    "name",
    [
        "gaia_level_studies",
        "android_env_studies",
        "api_bank_studies",
        "voyager_minecraft_studies",
        "web_shopping_studies",
        "video_game_studies",
    ],
)
def test_w1337_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
