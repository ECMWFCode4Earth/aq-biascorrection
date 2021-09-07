import datetime
from pathlib import Path, PosixPath

import pandas as pd
from cdsapi.api import Client

from src.constants import ROOT_DIR
from src.data.forecast import CAMSProcessor
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
        date: datetime.datetime,  # Date to make the predictions
        model: str,  # Model name
        forecast_idir: PosixPath,  # Input directory where the last day and following
        # day data are available
        intermediary_dir: PosixPath,
        output_dir: PosixPath,
        stations_csv: PosixPath,
        station_id: str,
    ):
        start_date = datetime.datetime.strftime(
            date - datetime.timedelta(days=1), "%Y-%m-%d"
        )
        end_date = datetime.datetime.strftime(date, "%Y-%m-%d")
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
        self.location = Location(**station.iloc[0].to_dict())
        self.stations_csv = stations_csv
        self.intermediary_dir = intermediary_dir
        self.output_dir = output_dir
        self.api_download_forecast = Client()

    def run(self):
        self.download_date_and_previous_date()
        CAMSProcessor(
            input_dir=self.forecast_idir,
            intermediary_dir=self.intermediary_dir,
            locations_csv=self.stations_csv,
            output_dir=self.output_dir,
            time_range=self.time_range,
        ).run_one_station(self.location)
        forecast_path = self.location.get_forecast_path(
            self.output_dir, "_".join(self.time_range.values()).replace("-", "")
        )
        return None

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
        leadtimes = [str(x) for x in list(range(0, 24))]
        for date in pd.date_range(
            self.time_range["start"], self.time_range["end"], freq="1D"
        ):
            date_str = datetime.datetime.strftime(date, "%Y-%m-%d")
            for variable, abbreviation in variables_to_abreviation.items():
                for leadtime in leadtimes:
                    download_path = (
                        self.forecast_idir / f"z_cams_c_ecmf_"
                        f"{date_str}_{leadtime}_{abbreviation}.nc"
                    )
                    self.api_download_forecast.retrieve(
                        "cams-global-atmospheric-composition-forecasts",
                        {
                            "variable": variable,
                            "date": date_str,
                            "time": "00:00",
                            "leadtime_hour": leadtime,
                            "type": "forecast",
                            "format": "netcdf_zip",
                        },
                        download_path,
                    )


if __name__ == "__main__":
    NearRealTimeWorkflow(
        date=datetime.datetime(year=2021, month=8, day=20),
        model="inception_time",
        forecast_idir=Path("/data2/cams_last_data"),
        intermediary_dir=Path("/tmp"),
        output_dir=Path("/tmp"),
        stations_csv=ROOT_DIR / "data/external/stations.csv",
        station_id="ES002",
    ).run()
