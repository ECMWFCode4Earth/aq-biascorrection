import pandas as pd

from pytest_mock import MockerFixture

from src.data import transformation_location
from src.data.utils import Location
from src.constants import ROOT_DIR


def test_location_transformation(mocker: MockerFixture):
    mocker.patch.object(
        transformation_location.Location,
        "get_observations_path",
        return_value=ROOT_DIR / "tests/data/pm25_test_obs.nc",
    )
    mocker.patch.object(
        transformation_location.Location,
        "get_forecast_path",
        return_value=ROOT_DIR / "tests/data/cams_tests.nc",
    )
    loc = Location(
        "TEST01", "Santander", "country_test", 40.4165, -3.70256, "Europe/Madrid", 668
    )
    lt = transformation_location.LocationTransformer(
        "pm25",
        loc,
        ROOT_DIR / "tests/data/observations/",
        ROOT_DIR / "tests/data/forecasts/",
    )
    results = lt.run()
    assert isinstance(results, pd.DataFrame)
    assert results["pm25_observed"].min() > 0
    assert results["pm25_observed"].notna().any()
