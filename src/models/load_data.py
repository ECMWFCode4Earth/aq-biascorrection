from pathlib import Path
from src.data.utils import Location
from typing import Dict

import xarray as xr


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