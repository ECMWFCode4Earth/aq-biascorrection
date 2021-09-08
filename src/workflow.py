import datetime
import os
import zipfile
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import concurrent.futures
from cdsapi.api import Client
from matplotlib.dates import DateFormatter

from src.constants import ROOT_DIR
from src.data.forecast import CAMSProcessor
from src.data.transformer import LocationTransformer
from src.data.utils import Location
from src.models.predict import ModelPredict
from src.models.validation import ValidationDataset


class Workflow:
    def __init__(
        self,
        variable: str,
        date: datetime.datetime,  # Date to make the predictions
        model: Path,  # Model name
        data_dir: Path,  # Input directory where all the steps will be performed
        stations_csv: Path,
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
        self.data_dir = data_dir / end_date
        self.download_dir = self.data_dir / "download"
        self.forecasts_dir = self.data_dir / "forecasts"
        self.observations_dir = self.data_dir / "observations"
        self.processed_dir = self.data_dir / "processed"
        self.predictions_dir = self.data_dir / "predictions"
        for directory in [
            self.data_dir,
            self.download_dir,
            self.forecasts_dir,
            self.observations_dir,
            self.processed_dir,
            self.predictions_dir,
        ]:
            if not directory.exists():
                os.makedirs(directory, exist_ok=True)
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
        for var in ["longitude", "latitude", "elevation"]:
            dict_to_location[var] = float(dict_to_location[var])
        self.location = Location(**dict_to_location)
        self.stations_csv = stations_csv
        self.intermediary_dir = Path('/tmp')
        self.api_download_forecast = Client(
            url="https://ads.atmosphere.copernicus.eu/api/v2",
            key="6858:5edcc1e8-e2c6-463b-8b18-d4ea2bafa965",
        )

    def run(self):
        self.download_forecast_data()
        CAMSProcessor(
            input_dir=self.download_dir,
            intermediary_dir=self.intermediary_dir,
            locations_csv=self.stations_csv,
            output_dir=self.forecasts_dir,
            time_range=self.time_range,
        ).run_one_station(self.location)
        observations_path = self.location.get_observations_path(
            self.observations_dir,
            self.variable,
            "_".join(self.time_range.values()).replace("-", ""),
        )
        if not observations_path.parent.exists():
            os.makedirs(observations_path.parent, exist_ok=True)
        self.create_observations_fake_dataset(observations_path)
        data = LocationTransformer(
            self.variable,
            self.location,
            self.observations_dir,
            self.forecasts_dir,
            self.time_range,
        ).run()
        processed_path = Path(
            self.processed_dir,
            self.variable,
            f"data_{self.variable}_{self.location.location_id}.csv",
        )
        if not processed_path.parent.exists():
            os.makedirs(processed_path.parent, exist_ok=True)
        data.to_csv(processed_path)
        predictions_paths = ModelPredict(
            config_yml_filename=self.model,
            predictions_dir=self.predictions_dir,
            input_data_dir=self.processed_dir,
        ).run()

        predictions, cams_and_obs = self.load_dataset(predictions_paths)
        val_data = self.get_initialization_datasets(predictions, cams_and_obs)
        df_final = val_data.cams.join([val_data.predictions])
        df_final.to_csv(
            self.data_dir / f"{self.variable}_predictions_"
            f'{self.location.location_id}_{self.time_range["end"]}.csv'
        )
        self.plot_time_serie(df_final)
        return df_final

    def download_forecast_data(self):
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
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future_to_entry = {
                        executor.submit(
                            self._download_forecast_data,
                            leadtime,
                            date_str,
                            abbreviation,
                            variable,
                            date
                        ): leadtime
                        for leadtime in leadtimes
                    }
                    for future in concurrent.futures.as_completed(future_to_entry):
                        result = future.result()

    def _download_forecast_data(
            self,
            leadtime,
            date_str,
            abbreviation,
            variable,
            date
    ):
        if len(leadtime) == 1:
            leadtime_str = f"00{leadtime}"
        elif len(leadtime) == 2:
            leadtime_str = f"0{leadtime}"
        else:
            leadtime_str = leadtime
        download_path = (
            self.download_dir / f"z_cams_c_ecmf_"
            f"{date_str}_fc_{leadtime_str}_{abbreviation}.zip"
        )
        download_path_nc = (
            self.download_dir / f"z_cams_c_ecmf_"
            f"{date_str}_fc_{leadtime_str}_{abbreviation}.nc"
        )
        if not download_path_nc.exists():
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
            with zipfile.ZipFile(download_path, "r") as zp:
                zp.extractall(download_path.parent)
                os.rename(download_path.parent / "data.nc", download_path_nc)
            os.remove(download_path)

    def create_observations_fake_dataset(self, observations_path):
        times = pd.date_range(
            f"{self.time_range['start']} 00:00",
            f"{self.time_range['end']} 23:59",
            freq="1H",
        )
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
                    "data": [values],
                }
            },
        }
        ds = xr.Dataset.from_dict(ds_dict)
        ds.to_netcdf(observations_path)

    def load_dataset(self, prediction_paths):
        # Load obs and cams:
        data_file = list(
            self.processed_dir.rglob(
                f"data_{self.variable}_{self.location.location_id}.csv"
            )
        )[0]
        data = pd.read_csv(data_file, index_col=0)
        data["index"] = pd.to_datetime(data["index"])
        obs_and_cams = data.set_index("index")

        # Load predictions:
        predictions, count = None, 0
        for prediction_path in prediction_paths:
            count += 1
            df = pd.read_csv(prediction_path, index_col=[0, 1])
            if predictions is not None:
                predictions += df
            else:
                predictions = df
        predictions /= count
        return predictions, obs_and_cams

    def get_initialization_datasets(
        self, df: pd.DataFrame, data: pd.DataFrame
    ) -> ValidationDataset:
        """
        Method to transform the 24 machine learning predictions columns into a single
        column dataframe with the temporal data.
        Args:
            df: machine learning predictions for correcting the CAMS forecast
            data: CAMS forecast data and observations
        """
        init_datasets = []
        for init_time, values in df.iterrows():
            indices = pd.date_range(start=init_time[0], periods=len(values), freq="H")
            # Perform the correction of the forecasts
            predictions = data.loc[indices, f"{self.variable}_forecast"] - values.values
            predictions = predictions.to_frame("Corrected CAMS").astype(float)
            cams = (
                data[f"{self.variable}_forecast"]
                .loc[predictions.index]
                .to_frame("CAMS")
                .astype(float)
            )
            obs = (
                data[f"{self.variable}_observed"]
                .loc[predictions.index]
                .to_frame("Observations")
                .astype(float)
            )
            persistence = (
                data[f"{self.variable}_observed"]
                .loc[predictions.index - datetime.timedelta(hours=24)]
                .reset_index(drop=True)
                .to_frame("Persistence")
                .set_index(predictions.index)
                .astype(float)
            )
            init_datasets.append(
                ValidationDataset(cams, obs, predictions, persistence, "test")
            )
        return init_datasets[0]

    def plot_time_serie(self, data):
        city = "".join(self.location.city.split(" ")).lower()
        country = "".join(self.location.country.split(" ")).lower()
        station_code = "".join(self.location.location_id.split(" ")).lower()
        filename = (
            self.data_dir
            / f"{self.variable}_timeserie_{station_code}_{city}_{country}.png"
        )
        colors = ["k", "red"]
        date_form = DateFormatter("%-d %b %H:%M")
        plt.figure(figsize=(30, 15))
        for i, column in enumerate(data.columns):
            plt.plot(
                data.index.values,
                data[column].values,
                linewidth=3,
                color=colors[i],
                label=column,
            )
        plt.legend()
        plt.ylabel(self.variable + r" ($\mu g / m^3$)", fontsize="xx-large")
        plt.xlabel("Date", fontsize="x-large")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(date_form)
        plt.title(
            f"{self.location.city} ({self.location.country})", fontsize="xx-large"
        )
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()


if __name__ == "__main__":
    for day in range(1, 32):
        time_0 = datetime.datetime.utcnow()
        Workflow(
            variable="no2",
            date=datetime.datetime(year=2021, month=8, day=day),
            model=Path("/home/pereza/datos/cams") / "config_inceptiontime_depth6.yml",
            data_dir=Path("/home/pereza/datos/cams"),
            stations_csv=ROOT_DIR / "data/external/stations.csv",
            station_id="ES002",
        ).run()
        time_1 = datetime.datetime.utcnow()
        total_time = time_1 - time_0
        print(total_time.total_seconds())
