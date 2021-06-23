from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytz
import logging
import xarray as xr

from src.data.utils import Location
from src.constants import ROOT_DIR


logger = logging.getLogger('Location Transformer')


def forecast_accumulated_variables_disaggregation(forecast_data):
    vars_to_temp_diss = ['dsrp', 'tp', 'uvb']
    for variable in vars_to_temp_diss:
        ds_variable = forecast_data[variable].copy()
        ds_variable_diff = ds_variable.differentiate(
            'time', 1, 'h'
        )
        ds_variable_diff = ds_variable_diff.where(
            ds_variable_diff >= 0,
            0
        )
        forecast_data[variable] = ds_variable_diff
    return forecast_data


class LocationTransformer:
    def __init__(
            self,
            variable: str,
            location: Location,
            observations_dir: Path = ROOT_DIR / 'data/interim/observations/',
            forecast_dir: Path = ROOT_DIR / 'data/interim/forecasts/',
            time_range: Dict[str, str] = None
    ):
        self.variable = variable
        if time_range is None:
            time_range = dict(start='2019-06-01', end='2021-03-31')
        self.time_range = time_range
        self.location = location
        self.observations_path = location.get_observations_path(
            observations_dir,
            self.variable,
            '_'.join(
                self.time_range.values()
            ).replace('-', '')
        )
        self.forecast_path = location.get_forecast_path(
            forecast_dir,
            '_'.join(
                self.time_range.values()
            ).replace('-', '')
        )

    def run(self) -> pd.DataFrame:
        # Open forecast and observational data
        forecast_data = self.opening_and_transforming_forecast()
        try:
            observed_data = self.opening_and_transforming_observations()
        except Exception as ex:
            raise Exception('There is not data for this variable at the'
                            ' location of interest')
        # Merge both xarray datasets
        merged = xr.merge([forecast_data, observed_data])
        merged_pd = merged.to_dataframe()
        # Adding local_time as a coordinate
        merged_pd = self.adding_local_time_hour(merged_pd)
        # There are some stations which has 0s, which seems to be NaN, drop
        # them
        varname = f'{self.variable}_observed'
        merged_pd[varname] = merged_pd[varname].where(merged_pd[varname] > 0)
        # There are sometimes where the observation is NaN, we drop these values
        merged_pd = merged_pd.dropna()
        # Calculation of the bias
        merged_pd[
            f'{self.variable}_bias'
        ] = merged_pd[
            f'{self.variable}_forecast'
        ] - merged_pd[
            f'{self.variable}_observed'
        ]
        merged_pd.reset_index(inplace=True)
        return merged_pd

    def opening_and_transforming_forecast(self) -> xr.Dataset:
        # Open the data
        forecast_data = xr.open_dataset(self.forecast_path)
        #Rename some of the variables
        forecast_data = forecast_data.rename({'pm2p5': 'pm25',
                                              'go3': 'o3'})
        # Interpolate time axis to 1h data
        logger.info("Interpolating time data to hourly resolution.")
        hourly_times = pd.date_range(forecast_data.time.values[0],
                                  forecast_data.time.values[-1],
                                  freq='1H')
        forecast_data = forecast_data.interp(time=hourly_times,
                                             method='linear')

        # Transform units of concentration variables
        for variable in ['pm25', 'o3', 'no2', 'so2', 'pm10']:
            logger.info(f"Transforming data for variable {variable}.")
            # The air density depends on temperature and pressure, but an
            # standard is known when 15K and 1 atmosphere of pressure
            surface_pressure = self.calculate_surface_pressure_by_msl(
                forecast_data['t2m'],
                forecast_data['msl']
            )
            air_density = self.calculate_air_density(
                surface_pressure,
                forecast_data['t2m']
            )
            # Now, we use the air density to transform to Micrograms / m³
            forecast_data[variable] *= air_density.values
            forecast_data[variable] *= (10 ** 9)

        # Some forecast variables are aggregated daily, so a temporal
        # disaggregation is needed
        logger.info(f"Dissaggregate forecast variables.")
        forecast_data = forecast_accumulated_variables_disaggregation(
            forecast_data
        )
        # Rename all the variables to "{variable}_forecast" in order to
        # distinguish them when merged
        for data_var in list(forecast_data.data_vars.keys()):
            forecast_data = forecast_data.rename(
                {data_var: f"{data_var}_forecast"}
            )

        forecast_data = forecast_data.drop(
            ['latitude', 'longitude', 'station_id']
        )
        return forecast_data

    def opening_and_transforming_observations(self) -> xr.Dataset:
        """
        Open the observations given the path specified in the object 
        declaration. It also transgforms the units of the air quality variables
        and filter the outliers.

        Returns:
            xr.Dataset: the observations dataset
        """
        # Open the data
        observations_data = xr.open_dataset(self.observations_path)
        # The variable 'o3' is in units of 'ppm' for the observations
        # which corresponds with the same as Miligrams / Kilogram,
        # we want to transform it to micrograms / m³
        if self.variable == 'o3':
            observations_data[self.variable] *= 10 ** 3
            # The air density depends on temperature and pressure, but an
            # standard is known when 15K and 1 atmosphere of pressure
            air_density = 0.816
            # Now, we use the air density to transform to Micrograms / m³
            observations_data[self.variable] /= air_density
        # Resample the values in order to have the same time frequency as
        # CAMS model forecast
        observations_data = observations_data.resample(
            {'time': '1H'}
        ).mean('time')
        # If there are more than one station associated with the location of
        # interest an average is performed taking into consideration the
        # distance to the location of interest
        observations_data = self.weight_average_with_distance(observations_data)

        # Rename all the variables to "{variable}_forecast" in order to
        # distinguish them when merged
        for data_var in list(observations_data.data_vars.keys()):
            observations_data = observations_data.rename(
                {data_var: f"{data_var}_observed"}
            )
        # Filter outliers
        observations_data = self.filter_observations_data(observations_data)

        return observations_data
    
    def filter_observations_data(
        self, 
        data: xr.Dataset, 
        rate_iqr: float = 20
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
        q3 = float(data.quantile(0.75)[f'{self.variable}_observed'])
        q1 = float(data.quantile(0.25)[f'{self.variable}_observed'])
        iqr = q3 - q1
        thres = q3 + rate_iqr * iqr

        # Filter values over the specified threshold
        logger.debug(f"Filtering observations values over {thres:.2f}.")
        filtered_data = data.where(data[f'{self.variable}_observed'] < thres)
        return filtered_data.dropna('time')

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
            ds = ds.mean('station_id')
            ds = ds.drop(['x', 'y', '_x', '_y', 'distance'])
        else:
            values_weighted_average = []
            for time in ds.time.values:
                ds_time = ds.sel(time=time)
                distance_and_value = {}
                for station in ds_time.station_id.values:
                    ds_station = ds_time.sel(station_id=station)
                    if ds_station.distance.values > 1:
                        distance_weight = round(1 / ds_station.distance.values, 2)
                    else:
                        # We give the same weight to those stations 1km close
                        # to the location of interest
                        distance_weight = 1
                    value = float(ds_station[self.variable].values)
                    if not np.isnan(value):
                        distance_and_value[distance_weight] = value
                if len(distance_and_value) == 0:
                    values_weighted_average.append(np.nan)
                else:
                    weights_normalized = np.array(
                        list(distance_and_value.keys())
                    ) / sum(distance_and_value.keys())
                    values_weighted_average.append(
                        np.average(list(distance_and_value.values()),
                                   weights=weights_normalized)
                    )
            ds = ds.drop(['x', 'y', '_x', '_y', 'distance'])
            ds = ds.mean('station_id')
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
        timezone = pytz.timezone(
            self.location.timezone
        )
        local_time_hour = [
            timezone.fromutc(
                pd.to_datetime(x)
            ).hour for x in df.index.values
        ]
        df['local_time_hour'] = local_time_hour
        return df

    def calculate_surface_pressure_by_msl(
        self,
        temp: xr.DataArray,
        mslp: xr.DataArray
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

    def calculate_air_density(
        self,
        pressure: xr.DataArray,
        temperature: xr.DataArray
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
