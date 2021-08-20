import pathlib

import pytest

from src.data.openaq_obs import OpenAQDownloader
from src.data.utils import Location, get_elevation_for_location


class TestOpenAQDownload:
    @pytest.fixture()
    def mocked_download_obj(self):
        args = dict(
            location_id="ES",
            city="Sebastopol",
            country="Rusia",
            latitude=37.9,
            longitude=39.8,
            timezone="Asia/China",
            elevation=10.5,
        )
        location = Location(**args)
        openaq_obj = OpenAQDownloader(
            location,
            pathlib.Path("/tmp"),
            "no2",
            dict(start="2019-06-01", end="2021-03-31"),
        )
        return openaq_obj

    def test_openaq_station_get_distance_to_real_point(self):
        distance = OpenAQDownloader.get_distance_between_two_points_on_earth(
            10, 11, 10, 11
        )
        assert type(distance) == float
        assert round(distance) == 156



class TestUtils:
    @pytest.fixture()
    def mocked_location(self):
        args = dict(
            location_id="ES",
            city="Sebastopol",
            country="Rusia",
            latitude=37.9,
            longitude=39.8,
            timezone="Asia/China",
            elevation=10.5,
        )
        location = Location(**args)
        return location

    def test_location_string_method(self, mocked_location):
        location = mocked_location
        string_to_check = (
            f"Location(location_id={location.location_id}, "
            f"city={location.city}, country={location.country}, "
            f"latitude={location.latitude}, longitude={location.longitude}, "
            f"elevation={location.elevation}, timezone={location.timezone}"
        )
        assert str(location) == string_to_check

    def test_location_get_observations_path(self, mocked_location):
        location = mocked_location
        observation_path = location.get_observations_path(
            "/tmp", "no2", "20190101-20210331"
        )
        args_from_path = str(observation_path).split("/")
        assert args_from_path[2] == location.country.lower().replace(" ", "-")
        assert args_from_path[3] == location.city.lower().replace(" ", "-")
        assert args_from_path[4] == location.location_id.lower()
        assert args_from_path[5] == "no2"
        assert type(observation_path) == pathlib.PosixPath

    def test_location_get_forecasts_path(self, mocked_location):
        location = mocked_location
        forecast_path = location.get_forecast_path("/tmp", "20190101-20210331")
        args_from_path = str(forecast_path).split("/")
        assert args_from_path[2] == location.country.lower().replace(" ", "-")
        assert args_from_path[3] == location.city.lower().replace(" ", "-")
        assert args_from_path[4] == location.location_id.lower()
        assert type(forecast_path) == pathlib.PosixPath

    def test_location_get_location_by_id(self, location_id="ES001"):
        location = Location.get_location_by_id(location_id)
        assert type(location) == Location
        assert location.country == "Spain"
        assert location.city == "Madrid"

    def test_location_get_elevation(self, mocked_location):
        location = mocked_location
        elevation = get_elevation_for_location(location.latitude, location.longitude)
        assert type(elevation) == int
        assert elevation >= 0
