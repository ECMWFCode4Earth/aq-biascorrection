from pathlib import Path
from src.data.utils import Location
from typing import Dict

import xarray as xr
import pandas as pd
import numpy as np
import pytz


class DataLoader:
    def __init__(
            self,
            variable: str,
            location: Location,
            observations_dir: Path = Path('../../data/processed/observations/'),
            forecast_dir: Path = Path('../../data/processed/forecasts/'),
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

    def run(self):
        # Open forecast and observational data
        forecast_data = self.opening_and_transforming_forecast()
        try:
            observed_data = self.opening_and_transforming_observations()
        except:
            raise Exception('There is not data for this variable at the'
                            ' location of interest')
        # Merge both xarray datasets
        merged = xr.merge([forecast_data, observed_data])
        merged_pd = merged.to_dataframe()
        # Adding local_time as a coordinate
        merged_pd = self.adding_local_time_hour(merged_pd)
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
        return merged_pd

    def opening_and_transforming_forecast(self):
        # Open the data
        forecast_data = xr.open_dataset(self.forecast_path)
        #Rename some of the variables
        forecast_data = forecast_data.rename({'pm2p5': 'pm25',
                                              'go3': 'o3'})
        # Transform units of concentration variables
        for variable in ['pm25', 'o3', 'no2', 'so2', 'pm10']:
            # Micrograms / kilogram
            forecast_data[variable] *= (10**9)
            # The air density depends on temperature and pressure, but an
            # standard is known when 15K and 1 atmosphere of pressure
            air_density = 0.816
            # Now, we use the air density to transform to Micrograms / m³
            forecast_data[variable] /= air_density

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

    def opening_and_transforming_observations(self):
        # Open the data
        observations_data = xr.open_dataset(self.observations_path)
        # The variable 'o3' is in units of 'ppm' for the observations
        # which corresponds with the same as Miligrams / Kilogram,
        # we want to transform it to micrograms / m³
        if self.variable == 'o3':
            observations_data[self.variable] *= 10**3
            # The air density depends on temperature and pressure, but an
            # standard is known when 15K and 1 atmosphere of pressure
            air_density = 0.816
            # Now, we use the air density to transform to Micrograms / m³
            observations_data[self.variable] /= air_density
        # Resample the values in order to have the same time frequency as
        # CAMS model forecast
        observations_data = observations_data.resample(
            {'time': '3H'}
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
        return observations_data

    def weight_average_with_distance(self, ds):
        values_weighted_average = []
        for time in ds.time.values:
            ds_time = ds.sel(time=time)
            distance_and_value = {}
            for station in ds_time.station_id.values:
                ds_station = ds_time.sel(station_id=station)
                distance_weight = round(1 / ds_station.distance.values, 2)
                value = float(ds_station[self.variable].values)
                if not np.isnan(value):
                    distance_and_value[distance_weight] = value
            if len(distance_and_value) == 0:
                values_weighted_average.append(np.nan)
            else:
                values_weighted_average.append(
                    np.average(list(distance_and_value.values()),
                               weights=list(distance_and_value.keys()))
                )
        ds = ds.mean('station_id')
        ds = ds.drop(['x', 'y', '_x', '_y', 'distance'])
        ds[self.variable][:] = values_weighted_average
        return ds

    def adding_local_time_hour(self, df):
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


if __name__ == '__main__':
    latitude_dubai = 25.0657
    longitude_dubai = 55.17128
    timezone = "Asia/Dubai"
    DataLoader(
        'pm25',
        Location(
            "AE001", "Dubai", "United Arab Emirates",
            latitude_dubai, longitude_dubai, timezone),
    ).run()
