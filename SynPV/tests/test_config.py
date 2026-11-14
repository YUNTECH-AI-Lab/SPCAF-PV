import pytest

from synpv.config import DEFAULT_CONFIG, default_config, paper_config, prevailing_wind_from, timezone_for


def test_defaults_are_copies():
    a = default_config()
    a["N_HOUSES"] = 3
    assert DEFAULT_CONFIG["N_HOUSES"] == 100


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        default_config(NOT_A_KEY=1)


def test_list_to_tuple():
    assert default_config(TILT_RANGE=[10, 30])["TILT_RANGE"] == (10, 30)


def test_paper_preset():
    c = paper_config()
    assert (c["LATITUDE"], c["N_HOUSES"], c["DESIGN_T_MIN"], c["CLOUD_PREVAILING_FROM_DEG"]) == (40.7259, 100, -18.0, 260.0)


@pytest.mark.parametrize("lat,expected", [(40.7, 260.0), (-35, 280.0), (10, 70.0), (-10, 110.0), (70, 80.0)])
def test_prevailing_wind(lat, expected):
    assert prevailing_wind_from(lat) == expected


def test_timezone():
    assert timezone_for(40.7259, -73.5143) == "America/New_York"
    assert timezone_for(-33.87, 151.21) == "Australia/Sydney"
