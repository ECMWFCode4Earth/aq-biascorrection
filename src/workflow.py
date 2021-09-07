import datetime
import os
import zipfile
from pathlib import Path, PosixPath

import numpy as np
import pandas as pd
import xarray as xr
from cdsapi.api import Client

from src.constants import ROOT_DIR
from src.data.forecast import CAMSProcessor
from src.data.transformer import LocationTransformer
from src.data.utils import Location
from src.models.config.gradient_boosting import GradientBoosting
from src.models.config.inception_time import InceptionTime
from src.models.config.regression import ElasticNetRegr

models_dict = {
    "gradient_boosting": GradientBoosting,
    "inception_time": InceptionTime,
    "elasticnet_regressor": ElasticNetRegr,
}


class NearRealTimeWorkflow:
    def __init__(
        self,
        variable: str,
        date: datetime.datetime,  # Date to make the predictions
        model: str,  # Model name
        forecast_idir: PosixPath,  # Input directory where the last day and following
        # day data are available
        intermediary_dir: PosixPath,
        output_dir: PosixPath,
        stations_csv: PosixPath,
        station_id: str,
    ):
        self.variable = variable
        start_date = datetime.datetime.strftime(
            date - datetime.timedelta(days=1), "%Y-%m-%d"
        )
        end_date = datetime.datetime.strftime(date, "%Y-%m-%d")
        self.end_date_for_download = datetime.datetime.strftime(
            date + datetime.timedelta(days=1), "%Y-%m-%d"
        )
        self.time_range = dict(start=start_date, end=end_date)
        self.model = model
        self.forecast_idir = forecast_idir
        stations = pd.read_csv(
            stations_csv,
            index_col=0,
            names=[
                "location_id",
                "city",
                "country",
                "latitude",
                "longitude",
                "timezone",
                "elevation",
            ],
        )
        station = stations[stations["location_id"] == station_id]
        dict_to_location = station.iloc[0].to_dict()
        for var in ['longitude', 'latitude', 'elevation']:
            dict_to_location[var] = float(dict_to_location[var])
        self.location = Location(**dict_to_location)
        self.stations_csv = stations_csv
        self.intermediary_dir = intermediary_dir
        self.output_dir = output_dir
        self.api_download_forecast = Client(
            url='https://ads.atmosphere.copernicus.eu/api/v2',
            key='6858:5edcc1e8-e2c6-463b-8b18-d4ea2bafa965'
        )

    def run(self):
        self.download_date_and_previous_date()
        CAMSProcessor(
            input_dir=self.forecast_idir,
            intermediary_dir=self.intermediary_dir,
            locations_csv=self.stations_csv,
            output_dir=self.forecast_idir / 'forecasts',
            time_range=self.time_range,
        ).run_one_station(self.location)
        forecast_path = self.location.get_forecast_path(
            self.forecast_idir / 'forecasts',
            "_".join(self.time_range.values()).replace("-", "")
        )
        observations_path = self.location.get_observations_path(
            self.forecast_idir / 'observations',
            self.variable,
            "_".join(self.time_range.values()).replace("-", "")
        )
        self.create_observations_fake_dataset(observations_path)
        data = LocationTransformer(
            self.variable,
            self.location,
            self.forecast_idir / 'observations',
            self.forecast_idir / 'forecasts',
            self.time_range
        ).run()
        return data

    def download_date_and_previous_date(self):
        variables_to_abreviation = {
            "10m_u_component_of_wind": "10u",
            "10m_v_component_of_wind": "10v",
            "2m_dewpoint_temperature": "2d",
            "2m_temperature": "2t",
            "boundary_layer_height": "blh",
            "direct_solar_radiation": "dsrp",
            "downward_uv_radiation_at_the_surface": "uvb",
            "mean_sea_level_pressure": "msl",
            "nitrogen_dioxide": "no2conc",
            "ozone": "go3conc",
            "particulate_matter_10um": "pm10",
            "particulate_matter_2.5um": "pm2p5",
            "sulphur_dioxide": "so2conc",
            "surface_geopotential": "z",
            "total_cloud_cover": "tcc",
            "total_precipitation": "tp",
        }
        leadtimes = [str(x) for x in list(range(0, 24, 3))]
        for date in pd.date_range(
            self.time_range["start"], self.end_date_for_download, freq="1D"
        ):
            date_str = datetime.datetime.strftime(date, "%Y%m%d")
            for variable, abbreviation in variables_to_abreviation.items():
                for leadtime in leadtimes:
                    if len(leadtime) == 1:
                        leadtime_str = f'00{leadtime}'
                    elif len(leadtime) == 2:
                        leadtime_str = f'0{leadtime}'
                    else:
                        leadtime_str = leadtime
                    download_path = (
                        self.forecast_idir / f"z_cams_c_ecmf_"
                        f"{date_str}_fc_{leadtime_str}_{abbreviation}.zip"
                    )
                    download_path_nc = (
                            self.forecast_idir / f"z_cams_c_ecmf_"
                            f"{date_str}_fc_{leadtime_str}_{abbreviation}.nc"
                    )
                    if download_path_nc.exists():
                        continue
                    try:
                        self.api_download_forecast.retrieve(
                            "cams-global-atmospheric-composition-forecasts",
                            {
                                "variable": variable,
                                "date": datetime.datetime.strftime(date, "%Y-%m-%d"),
                                "time": "00:00",
                                "leadtime_hour": leadtime,
                                "type": "forecast",
                                "format": "netcdf_zip",
                            },
                            download_path,
                        )
                    except:
                        self.api_download_forecast.retrieve(
                            "cams-global-atmospheric-composition-forecasts",
                            {
                                "variable": variable,
                                "model_level": "137",
                                "date": datetime.datetime.strftime(date, "%Y-%m-%d"),
                                "time": "00:00",
                                "leadtime_hour": leadtime,
                                "type": "forecast",
                                "format": "netcdf_zip",
                            },
                            download_path,
                        )
                    with zipfile.ZipFile(download_path, 'r') as zp:
                        zp.extractall(download_path.parent)
                        os.rename(download_path.parent / 'data.nc',
                                  download_path_nc)
                    os.remove(download_path)

    def create_observations_fake_dataset(self, observations_path):
        times = pd.date_range(f"{self.time_range['start']} 00:00",
                              f"{self.time_range['end']} 23:59",
                              freq='1H')
        values = list(range(len(times)))
        ds_dict = {
            "coords": {
                "time": {
                    "dims": ("time",),
                    "attrs": {},
                    "data": times,
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
            "dims": {"station_id": 1, "time": len(times)},
            "data_vars": {
                "no2": {
                    "dims": ("station_id", "time"),
                    "attrs": {
                        "units": "microgram / m^3",
                        "standard_name": "no2",
                        "long_name": "Nitrogen dioxide",
                    },
                    "data": [
                        values
                    ],
                }
            },
        }
        ds = xr.Dataset.from_dict(ds_dict)
        if not observations_path.parent.exists():
            os.makedirs(observations_path.parent, exist_ok=True)
        ds.to_netcdf(observations_path)


if __name__ == "__main__":
    NearRealTimeWorkflow(
        variable='no2',
        date=datetime.datetime(year=2021, month=8, day=20),
        model="inception_time",
        forecast_idir=Path("/home/pereza/datos/cams"),
        intermediary_dir=Path("/tmp"),
        output_dir=Path("/tmp"),
        stations_csv=ROOT_DIR / "data/external/stations.csv",
        station_id="ES002",
    ).run()
