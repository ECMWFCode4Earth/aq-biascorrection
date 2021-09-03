import concurrent.futures
import logging
import os
import pathlib
from pathlib import Path
from typing import Dict, List

import pandas as pd
import xarray as xr
import numpy as np
import pytz

from src.constants import ROOT_DIR
from src.data.utils import Location
from src.logging import get_logger

logger = get_logger("Data Transformer")


class DataTransformer:
    def __init__(
        self,
        variable: str,
        locations_csv_path: Path = ROOT_DIR / "data/external/stations.csv",
        output_dir: Path = ROOT_DIR / "data/processed/",
        observations_dir: Path = ROOT_DIR / "data/interim/observations/",
        forecast_dir: Path = ROOT_DIR / "data/interim/forecasts/",
        time_range: Dict[str, str] = None,
    ):
        self.variable = variable
        self.locations = pd.read_csv(locations_csv_path)
        self.output_dir = output_dir
        self.observations_dir = observations_dir
        self.forecast_dir = forecast_dir
        if time_range is None:
            time_range = dict(start="2019-06-01", end="2021-03-31")
        self.time_range = time_range

    def run(self) -> List[Path]:
        data_for_locations_paths = []
        locations = [
            Location(
                location[1]["id"],
                location[1]["city"],
                location[1]["country"],
                location[1]["latitude"],
                location[1]["longitude"],
                location[1]["timezone"],
                location[1]["elevation"],
            )
            for location in self.locations.iterrows()
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_entry = {
                executor.submit(self._data_transform, location): location
                for location in locations
            }
            for future in concurrent.futures.as_completed(future_to_entry):
                result = future.result()
                if type(result) is pathlib.PosixPath:
                    logger.info(f"Intermediary data saved to: {result}")
                    data_for_locations_paths.append(result)
                else:
                    logger.error(result)
        return data_for_locations_paths

    def _data_transform(self, loc) -> Path or Exception:
        try:
            logger.info(f"Extracting data for location: {str(loc)}")
            inter_loc_path = self.get_output_path(loc)
            if inter_loc_path.exists():
                logger.info(f"Station at {loc.city} is already computed.")
                return inter_loc_path
            data_for_location = LocationTransformer(
                self.variable,
                loc,
                observations_dir=self.observations_dir,
                forecast_dir=self.forecast_dir,
                time_range=self.time_range,
            ).run()
            data_for_location.to_csv(str(inter_loc_path))
            return inter_loc_path
        except Exception as ex:
            return ex

    def get_output_path(self, loc: Location) -> Path:
        ext = ".csv"
        intermediary_path = Path(
            self.output_dir,
            self.variable,
            f"data_{self.variable}_{loc.location_id}{ext}",
        )
        if not intermediary_path.parent.exists():
            os.makedirs(intermediary_path.parent, exist_ok=True)
        return intermediary_path


class LocationTransformer:
    def __init__(
        self,
        variable: str,
        location: Location,
        observations_dir: Path = ROOT_DIR / "data/interim/observations/",
        forecast_dir: Path = ROOT_DIR / "data/interim/forecasts/",
        time_range: Dict[str, str] = None,
    ):
        self.variable = variable
        if time_range is None:
            time_range = dict(start="2019-06-01", end="2021-03-31")
        self.time_range = time_range
        self.location = location
        self.observations_path = location.get_observations_path(
            observations_dir,
            self.variable,
            "_".join(self.time_range.values()).replace("-", ""),
        )
        self.forecast_path = location.get_forecast_path(
            forecast_dir, "_".join(self.time_range.values()).replace("-", "")
        )

    def run(self) -> pd.DataFrame:
        """
        Main workflow for the LocationTransformer class

        Returns:
            pd.DataFrame: observations and forecasts merged
        """
        # Open forecast and observational data
        try:
            observed_data = self.opening_and_transforming_observations()
        except Exception as ex:
            raise Exception(
                "There is not data for this variable at the" " location of interest"
            )
        forecast_data = self.opening_and_transforming_forecast()
        # Merge both xarray datasets
        merged = xr.merge([forecast_data, observed_data])
        merged_pd = merged.to_dataframe()
        # Adding local_time as a coordinate
        merged_pd = self.adding_local_time_hour(merged_pd)
        # There are some stations which has 0s, which seems to be NaN, drop
        # them
        varname = f"{self.variable}_observed"
        merged_pd[varname] = merged_pd[varname].where(merged_pd[varname] > 0)
        # There are sometimes where the observation is NaN, we drop these values
        merged_pd = merged_pd.dropna()
        # Calculation of the bias
        merged_pd[f"{self.variable}_bias"] = (
            merged_pd[f"{self.variable}_forecast"]
            - merged_pd[f"{self.variable}_observed"]
        )
        merged_pd.reset_index(inplace=True)
        return merged_pd

    def opening_and_transforming_forecast(self) -> xr.Dataset:
        """
        Open the forecasts given the path specified in the object
        declaration. It also transforms the units of the air quality variables
        and disaggregate some variables temporally.

        Returns:
            xr.Dataset: the forecast dataset
        """
        # Open the data
        forecast_data = xr.open_dataset(self.forecast_path)
        # Rename some of the variables
        forecast_data = forecast_data.rename({"pm2p5": "pm25", "go3": "o3"})
        # Interpolate time axis to 1h data
        logger.info("Interpolating time data to hourly resolution.")
        hourly_times = pd.date_range(
            forecast_data.time.values[0], forecast_data.time.values[-1], freq="1H"
        )
        forecast_data = forecast_data.interp(time=hourly_times, method="linear")

        # Transform units of concentration variables
        for variable in ["pm25", "o3", "no2", "so2", "pm10"]:
            logger.info(f"Transforming data for variable {variable}.")
            # The air density depends on temperature and pressure, but an
            # standard is known when 15K and 1 atmosphere of pressure
            surface_pressure = self.calculate_surface_pressure_by_msl(
                forecast_data["t2m"], forecast_data["msl"]
            )
            air_density = self.calculate_air_density(
                surface_pressure, forecast_data["t2m"]
            )
            # Now, we use the air density to transform to Micrograms / m³
            forecast_data[variable] *= air_density.values
            forecast_data[variable] *= 10 ** 9

        # Some forecast variables are aggregated daily, so a temporal
        # disaggregation is needed
        logger.info(f"Dissaggregate forecast variables.")
        forecast_data = self.forecast_accumulated_variables_disaggregation(
            forecast_data
        )
        # Rename all the variables to "{variable}_forecast" in order to
        # distinguish them when merged
        for data_var in list(forecast_data.data_vars.keys()):
            forecast_data = forecast_data.rename({data_var: f"{data_var}_forecast"})

        forecast_data = forecast_data.drop(["latitude", "longitude", "station_id"])
        return forecast_data

    def opening_and_transforming_observations(self) -> xr.Dataset:
        """
        Open the observations given the path specified in the object
        declaration. It also transforms the units of the air quality variables
        and filter the outliers.

        Returns:
            xr.Dataset: the observations dataset
        """
        # Open the data
        observations_data = xr.open_dataset(self.observations_path)

        # Resample the values in order to have the same time frequency as
        # CAMS model forecast
        observations_data = observations_data.resample({"time": "1H"}).mean()
        # If there are more than one station associated with the location of
        # interest an average is performed taking into consideration the
        # distance to the location of interest
        observations_data = self.weight_average_with_distance(observations_data)

        # Rename all the variables to "{variable}_observed" in order to
        # distinguish them when merged
        for data_var in list(observations_data.data_vars.keys()):
            observations_data = observations_data.rename(
                {data_var: f"{data_var}_observed"}
            )
        # Filter outliers
        observations_data = self.filter_observations_data(observations_data)
        # Resample time axis to 1H time frequency
        observations_data = observations_data.resample({"time": "1H"}).asfreq()
        # Rolling through the data to interpolate NaNs
        for data_var in list(observations_data.data_vars.keys()):
            observations_data[data_var] = observations_data[data_var].interpolate_na(
                dim="time",
                method="linear",
                fill_value="extrapolate",
                use_coordinate=True,
                max_gap=pd.Timedelta(value=12, unit="h"),
            )
        return observations_data

    def filter_observations_data(
        self, data: xr.Dataset, rate_iqr: float = 20
    ) -> xr.Dataset:
        """
        Method for filtering extreme values for the air quality observed values.
        By defaults it filters values over 20 times the IQR over the third
        quartile.

        Args:
            data (xr.Dataset): dataset of observations to filter.
            rate_iqr (float): number of IQR to add to the third quartile.

        Returns:
            xr.Dataset: dataset containing the filtered observations.
        """
        q3 = float(data.quantile(0.75)[f"{self.variable}_observed"])
        q1 = float(data.quantile(0.25)[f"{self.variable}_observed"])
        iqr = q3 - q1
        thres = q3 + rate_iqr * iqr

        # Filter values over the specified threshold
        logger.debug(f"Filtering observations values over {thres:.2f}.")
        filtered_data = data.where(data[f"{self.variable}_observed"] < thres)
        return filtered_data.dropna("time")

    def weight_average_with_distance(self, ds: xr.Dataset) -> xr.Dataset:
        """
        This method calculates the value for the observational data as a weight
        average of the closes stations to the location of interest.

        Args:
            ds (xr.Dataset): dataset containing a dimension station_id, which
            represents different stations.

        Returns:
            xr.Dataset: dataset of observed value at the location specified.
        """
        if len(ds.station_id.values) == 1:
            ds = ds.mean("station_id")
            ds = ds.drop(["x", "y", "_x", "_y", "distance"])
        else:
            values_weighted_average = []
            for time in ds.time.values:
                ds_time = ds.sel(time=time)
                distances = ds_time.distance.values
                distances_weights = [
                    1 if distance <= 1 else round(1 / distance, 2)
                    for distance in distances
                ]
                values = ds_time[list(ds_time.data_vars)[0]].values
                assert len(distances_weights) == len(values)
                values_n = []
                distances_weights_n = []
                for i in range(len(values)):
                    if not np.isnan(values[i]):
                        values_n.append(values[i])
                        distances_weights_n.append(distances_weights[i])
                if len(values_n) == 0:
                    values_weighted_average.append(np.nan)
                else:
                    normalized_weights = [
                        weight / sum(distances_weights_n)
                        for weight in distances_weights_n
                    ]
                    values_weighted_average.append(
                        np.average(
                            values_n,
                            weights=normalized_weights,
                        )
                    )
            ds = ds.drop(["x", "y", "_x", "_y", "distance"])
            ds = ds.mean("station_id")
            ds[self.variable][:] = values_weighted_average
        return ds

    def adding_local_time_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method uses the Location object 'timezone' attribute to obtain the
        local time hour from the UTC time. This is importante step because the
        bias in the model is known to depend on the diurnal cycle (local time
        of the place is needed)

        Args:
            df (pd.DataFrame): table with index compatible with datetime.

        Returns:
            pd.DataFrame: table with new column containing the local time hour.
        """
        timezone = pytz.timezone(self.location.timezone)
        local_time_hour = [
            timezone.fromutc(pd.to_datetime(x)).hour for x in df.index.values
        ]
        df["local_time_hour"] = local_time_hour
        return df

    @staticmethod
    def forecast_accumulated_variables_disaggregation(forecast_data):
        vars_to_temp_diss = ["dsrp", "tp", "uvb"]
        for variable in vars_to_temp_diss:
            ds_variable = forecast_data[variable].copy()
            ds_variable_diff = ds_variable.differentiate("time", 1, "h")
            ds_variable_diff = ds_variable_diff.where(ds_variable_diff >= 0, 0)
            forecast_data[variable] = ds_variable_diff
        return forecast_data

    def calculate_surface_pressure_by_msl(
        self, temp: xr.DataArray, mslp: xr.DataArray
    ) -> xr.DataArray:
        """
        Method to get the surface pressure by using the elevation of the
        location of interest, the temperature (temp) and the mean sea
        level pressure (mslp).

        Args:
            temp (xr.DataArray): dataset of the temperature.
            mslp (xr.DataArray): dataset of the mean sea level pressure.

        Returns:
            xr.DataArray: dataset of the surface pressure
        """
        elevation = self.location.elevation
        exponent = (9.80665 * 0.0289644) / (8.31432 * 0.0065)
        factor = (1 + ((-0.0065 / temp) * elevation)) ** exponent
        surface_pressure = mslp * factor
        return surface_pressure

    @staticmethod
    def calculate_air_density(
        pressure: xr.DataArray, temperature: xr.DataArray
    ) -> xr.DataArray:
        """
        Method to get the air density at a given temperature and pressure.

        Args:
            pressure (xr.DataArray): dataset of pressure.
            temperature (xr.DataArray): dataset of temperature.

        Returns:
            xr.DataArray: dataset of air density.
        """
        return pressure / (temperature * 287.058)
