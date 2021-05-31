from pathlib import Path
from src.data.utils import Location
from typing import Dict

import xarray as xr
import pandas as pd
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
        forecast_data = xr.open_dataset(self.forecast_path)
        observations_data = xr.open_dataset(self.observations_path)
        observations_data = observations_data.resample(
            {'time': '3H'}
        ).mean('time')
        if len(observations_data.station_id.values) >= 1:
            observations_data = observations_data.mean(dim='station_id')
        forecast_and_obs = xr.merge([forecast_data, observations_data])
        timezone = pytz.timezone(self.location.timezone)
        forecast_and_obs['local_time'] = [
            timezone.fromutc(
                pd.to_datetime(x)
            ) for x in forecast_and_obs.time.values
        ]


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
