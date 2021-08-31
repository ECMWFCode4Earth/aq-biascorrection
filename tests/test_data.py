import datetime
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from mockito import ANY, mock, when

from src.data.openaq_obs import OpenAQDownloader
from src.data.utils import Location, get_elevation_for_location


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
        string_to_check = (
            f"Location(location_id={mocked_location.location_id}, "
            f"city={mocked_location.city}, "
            f"country={mocked_location.country}, "
            f"latitude={mocked_location.latitude}, "
            f"longitude={mocked_location.longitude}, "
            f"elevation={mocked_location.elevation}, "
            f"timezone={mocked_location.timezone}"
        )
        assert str(mocked_location) == string_to_check

    def test_location_get_observations_path(self, mocked_location):
        observation_path = mocked_location.get_observations_path(
            "/tmp", "no2", "20190101-20210331"
        )
        args_from_path = str(observation_path).split("/")
        assert args_from_path[2] == mocked_location.country.lower().replace(" ", "-")
        assert args_from_path[3] == mocked_location.city.lower().replace(" ", "-")
        assert args_from_path[4] == mocked_location.location_id.lower()
        assert args_from_path[5] == "no2"
        assert type(observation_path) == pathlib.PosixPath

    def test_location_get_forecasts_path(self, mocked_location):
        forecast_path = mocked_location.get_forecast_path("/tmp", "20190101-20210331")
        args_from_path = str(forecast_path).split("/")
        assert args_from_path[2] == mocked_location.country.lower().replace(" ", "-")
        assert args_from_path[3] == mocked_location.city.lower().replace(" ", "-")
        assert args_from_path[4] == mocked_location.location_id.lower()
        assert type(forecast_path) == pathlib.PosixPath

    def test_location_get_location_by_id(self, location_id="ES001"):
        location = Location.get_location_by_id(location_id)
        assert type(location) == Location
        assert location.country == "Spain"
        assert location.city == "Madrid"

    def test_location_get_elevation(self, mocked_location):
        elevation = get_elevation_for_location(
            mocked_location.latitude, mocked_location.longitude
        )
        assert type(elevation) == int
        assert elevation >= 0


class TestOpenAQDownload:
    @pytest.fixture()
    def mocked_download_obj(self):
        args = dict(
            location_id="ES002",
            city="Barcelona",
            country="Spain",
            latitude=41.39,
            longitude=2.16,
            timezone="Europe/Madrid",
            elevation=47,
        )
        location = Location(**args)
        openaq_obj = OpenAQDownloader(
            location=location,
            output_dir=pathlib.Path("/tmp"),
            variable="no2",
            time_range=dict(start="2019-06-01", end="2021-03-31"),
        )
        return openaq_obj

    @pytest.fixture()
    def mocked_output_path(self, tmp_path):
        tempdir = tmp_path / "sub"
        tempdir.mkdir()
        output_path = tempdir / "/output_file.nc"
        return output_path

    @pytest.fixture()
    def _mocked_dataframe_with_closest_stations(self):
        df_dict = {
            "id": {22: 3298},
            "city": {22: "Barcelona"},
            "name": {22: "ES1438A"},
            "entity": {22: "government"},
            "country": {22: "ES"},
            "sources": {
                22: [
                    {
                        "id": "eea",
                        "url": "http://www.eea.europa.eu/themes/air/air-quality",
                        "name": "EEA",
                    }
                ]
            },
            "isMobile": {22: False},
            "isAnalysis": {22: False},
            "parameters": {
                22: [
                    {
                        "id": 7189,
                        "unit": "µg/m³",
                        "count": 43299,
                        "average": 33.3172359638791,
                        "lastValue": 40,
                        "parameter": "no2",
                        "displayName": "NO₂ mass",
                        "lastUpdated": "2021-08-24T06:00:00+00:00",
                        "parameterId": 5,
                        "firstUpdated": "2016-11-17T23:00:00+00:00",
                    }
                ]
            },
            "sensorType": {22: "reference grade"},
            "lastUpdated": {22: pd.Timestamp("2021-08-24 06:00:00+0000", tz="UTC")},
            "firstUpdated": {22: pd.Timestamp("2016-11-17 23:00:00+0000", tz="UTC")},
            "measurements": {22: 161377},
            "coordinates.latitude": {22: 41.3853432834672},
            "coordinates.longitude": {22: 2.15382196},
        }
        df = pd.DataFrame(df_dict)
        return df

    @pytest.fixture()
    def mocked_dataframe_with_closest_stations(self):
        df_dict = {
            "id": {22: 3298},
            "city": {22: "Barcelona"},
            "name": {22: "ES1438A"},
            "entity": {22: "government"},
            "country": {22: "ES"},
            "sources": {
                22: [
                    {
                        "id": "eea",
                        "url": "http://www.eea.europa.eu/themes/air/air-quality",
                        "name": "EEA",
                    }
                ]
            },
            "isMobile": {22: False},
            "isAnalysis": {22: False},
            "parameters": {
                22: [
                    {
                        "id": 7189,
                        "unit": "µg/m³",
                        "count": 43299,
                        "average": 33.3172359638791,
                        "lastValue": 40,
                        "parameter": "no2",
                        "displayName": "NO₂ mass",
                        "lastUpdated": "2021-08-24T06:00:00+00:00",
                        "parameterId": 5,
                        "firstUpdated": "2016-11-17T23:00:00+00:00",
                    }
                ]
            },
            "sensorType": {22: "reference grade"},
            "lastUpdated": {22: pd.Timestamp("2021-08-24 06:00:00+0000", tz="UTC")},
            "firstUpdated": {22: pd.Timestamp("2016-11-17 23:00:00+0000", tz="UTC")},
            "measurements": {22: 161377},
            "coordinates.latitude": {22: 41.3853432834672},
            "coordinates.longitude": {22: 2.15382196},
            "distance": {22: 0.7308156936731197},
            "is_in_temporal_range": {22: 1.0},
            "combination_dist_and_is_in_temporal_range": {22: 0.7308156936731197},
        }
        df = pd.DataFrame(df_dict)
        return df

    @pytest.fixture()
    def mocked_dataset_with_closest_stations(self):
        ds_dict = {
            "coords": {
                "time": {
                    "dims": ("time",),
                    "attrs": {},
                    "data": [
                        datetime.datetime(2019, 6, 1, 8, 0),
                        datetime.datetime(2019, 6, 1, 10, 0),
                        datetime.datetime(2019, 6, 1, 11, 0),
                        datetime.datetime(2019, 6, 4, 16, 0),
                        datetime.datetime(2019, 6, 8, 2, 0),
                        datetime.datetime(2019, 6, 8, 3, 0),
                        datetime.datetime(2019, 6, 8, 10, 0),
                        datetime.datetime(2019, 6, 8, 11, 0),
                        datetime.datetime(2019, 6, 15, 10, 0),
                        datetime.datetime(2019, 6, 15, 11, 0),
                        datetime.datetime(2019, 6, 21, 1, 0),
                        datetime.datetime(2019, 6, 22, 8, 0),
                        datetime.datetime(2019, 6, 29, 8, 0),
                        datetime.datetime(2019, 7, 4, 13, 0),
                        datetime.datetime(2019, 7, 4, 14, 0),
                        datetime.datetime(2019, 7, 4, 15, 0),
                        datetime.datetime(2019, 7, 4, 16, 0),
                        datetime.datetime(2019, 7, 4, 17, 0),
                        datetime.datetime(2019, 7, 4, 18, 0),
                        datetime.datetime(2019, 7, 4, 19, 0),
                    ],
                },
                "station_id": {
                    "dims": ("station_id",),
                    "attrs": {"long_name": "station name", "cf_role": "timeseries_id"},
                    "data": [3298],
                },
                "x": {
                    "dims": (),
                    "attrs": {
                        "units": "degrees_east",
                        "long_name": "Longitude",
                        "standard_name": "longitude",
                    },
                    "data": 2.15382196,
                },
                "y": {
                    "dims": (),
                    "attrs": {
                        "units": "degrees_north",
                        "long_name": "Latitude",
                        "standard_name": "latitude",
                    },
                    "data": 41.3853432834672,
                },
                "_x": {
                    "dims": (),
                    "attrs": {
                        "units": "degrees_east",
                        "long_name": "Longitude of the location of interest",
                        "standard_name": "longitude_interest",
                    },
                    "data": 2.16,
                },
                "_y": {
                    "dims": (),
                    "attrs": {
                        "units": "degrees_north",
                        "long_name": "Latitude of the location of interest",
                        "standard_name": "latitude_interest",
                    },
                    "data": 41.39,
                },
                "distance": {
                    "dims": (),
                    "attrs": {
                        "units": "km",
                        "long_name": "Distance",
                        "standard_name": "distance",
                    },
                    "data": 0.7308156936731197,
                },
            },
            "attrs": {"featureType": "timeSeries", "Conventions": "CF-1.4"},
            "dims": {"station_id": 1, "time": 20},
            "data_vars": {
                "no2": {
                    "dims": ("station_id", "time"),
                    "attrs": {
                        "units": "microgram / m^3",
                        "standard_name": "no2",
                        "long_name": "Nitrogen dioxide",
                    },
                    "data": [
                        [
                            48,
                            43,
                            52,
                            60,
                            28,
                            26,
                            32,
                            27,
                            30,
                            30,
                            21,
                            26,
                            137,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ],
                }
            },
        }
        ds = xr.Dataset.from_dict(ds_dict)
        return ds

    def test_openaq_station_get_distance_to_real_point(self):
        distance = OpenAQDownloader.get_distance_between_two_points_on_earth(
            10, 11, 10, 11
        )
        assert type(distance) == float
        assert round(distance) == 156

    def test_openaq_download_run_with_existing_path(
        self, mocked_download_obj, mocked_output_path
    ):
        with open(mocked_output_path, "w") as output_file:
            output_file.write("Hi!")
        when(Location).get_observations_path(ANY(), ANY(), ANY()).thenReturn(
            mocked_output_path
        )
        result = mocked_download_obj.run()
        assert result == mocked_output_path

    def test_openaq_download_run_without_existing_path(
        self,
        mocked_download_obj,
        mocked_output_path,
        mocked_dataframe_with_closest_stations,
        mocked_dataset_with_closest_stations,
    ):
        when(Location).get_observations_path(ANY(), ANY(), ANY()).thenReturn(
            mocked_output_path
        )
        when(OpenAQDownloader).get_closest_stations_to_location().thenReturn(
            mocked_dataframe_with_closest_stations
        )
        when(OpenAQDownloader).get_data(ANY()).thenReturn(
            [mocked_dataset_with_closest_stations]
        )
        when(OpenAQDownloader).concat_filter_and_save_data(ANY(), ANY()).thenReturn(
            xr.concat([mocked_dataset_with_closest_stations], dim="station_id"),
        )
        result = mocked_download_obj.run()
        assert result == mocked_output_path

    def test_openaq_download_method_get_closest_stations(
        self,
        mocked_download_obj,
        _mocked_dataframe_with_closest_stations,
        mocked_dataframe_with_closest_stations,
    ):
        when(OpenAQDownloader)._get_closest_stations_to_location().thenReturn(
            _mocked_dataframe_with_closest_stations
        )
        stations = mocked_download_obj.get_closest_stations_to_location()
        assert len(stations) >= 0
        assert stations["distance"].values[0]
        assert stations["is_in_temporal_range"].values[0]
        assert stations["combination_dist_and_is_in_temporal_range"].values[0]
        assert np.all((stations == mocked_dataframe_with_closest_stations).values)

    def test_openaq_download_method__get_closest_stations(
        self, mocked_download_obj, _mocked_dataframe_with_closest_stations
    ):
        when(mocked_download_obj.api).locations(
            parameter=ANY(str),
            coordinates=ANY(str),
            radius=ANY(int),
            df=ANY(bool),
        ).thenReturn(_mocked_dataframe_with_closest_stations)
        stations = mocked_download_obj._get_closest_stations_to_location()
        assert len(stations) >= 0
