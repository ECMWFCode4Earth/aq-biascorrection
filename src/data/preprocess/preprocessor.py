from pathlib import Path
from src.data.utils import Location
from typing import Dict


import pandas as pd
import xarray as xr

import datetime
import logging
import glob


variables_dict = {
    "10u": "10 metre U wind component",
    "10v": "10 metre V wind component",
    "2d": "2 metre dewpoint temperature",
    "2t": "2 metre temperature",
    "blh": "Boundary layer height",
    "dsrp": "Direct solar radiation",
    "go3conc": "GEMS Ozone",
    "msl": "Mean sea level pressure",
    "no2conc": "Nitrogen dioxide",
    "pm10": "Particulate matter d < 10 um",
    "pm2p5": "Particulate matter d < 2.5 um",
    "so2conc": "Sulphur dioxide",
    "tcc": "Total cloud cover",
    "tp": "Total precipitation",
    "uvb": "Downward UV radiation at the surface",
    "z": "Geopotential"
}


class CAMSProcessor:
    """
    Class to process the CAMS model forecast
    """

    def __init__(
            self,
            input_dir: Path,
            location: Location,
            variable: str,
            output_dir: Path,
            time_range: Dict[str, str] = None
    ):
        self.input_dir = input_dir
        if time_range is None:
            self.time_range = dict(start='2019-06-01', end='2021-03-31')
        else:
            self.time_range = time_range

        if variable in ['o3', 'no2', 'so2', 'pm10', 'pm25']:
            self.variable = variable
        else:
            raise NotImplementedError(f"The variable {variable} do"
                                      f" not correspond to any known one")
        self.location = location
        self.output_dir = output_dir

    def run(self):
        initialization_times = self.get_initialization_times()
        total_data = []
        for initialization_time in initialization_times:
            try:
                paths_for_forecast = self.get_paths_for_forecasted_variables(
                    initialization_time
                )
                data = self.get_data(paths_for_forecast)
                total_data.append(data)
            except Exception as ex:
                logging.error(ex)
        total_data = xr.concat(total_data, dim='time')
        output_path = self.get_output_path()
        self.save_data(total_data, output_path)
        return total_data

    def get_initialization_times(self):
        initialization_times = pd.date_range(
            self.time_range['start'],
            self.time_range['end'],
            freq='1D'
        )
        initialization_times = [datetime.datetime.strftime(
            time, '%Y%m%d'
        ) for time in initialization_times]
        return initialization_times

    def get_paths_for_forecasted_variables(
            self,
            initialization_time: str
    ) -> list:
        ext = '.nc'
        input_path_pattern = Path(
            self.input_dir,
            f"z_cams_c_ecmf_{initialization_time}"
            f"_fc_*_*{ext}"
        )
        input_path_match = str(input_path_pattern)
        paths = glob.glob(input_path_match)
        paths = [path for path in paths if '_024_' not in path]
        if len(paths) == 0:
            raise Exception(f'There is no data for the initialization time:'
                            f' {initialization_time}')
        paths.sort()
        return paths

    def get_data(self, paths_for_forecast):
        data = xr.open_mfdataset(paths_for_forecast,
                                 concat_dim='time',
                                 preprocess=self.filter_location)
        return data

    def filter_location(self, data):
        data_location = data.sel(latitude=self.location.latitude,
                                 longitude=self.location.longitude,
                                 method='nearest')
        return data_location

    def get_output_path(self) -> Path:
        """
        Method to get the output paths where the data and metadata are stored.
        """
        city = self.location.city.lower().replace(' ', '_')
        country = self.location.country.lower().replace(' ', '_')
        station_id = self.location.location_id.lower()
        variable = self.variable
        time_range = '_'.join(
            self.time_range.values()
        ).replace('-', '')
        ext = '.nc'
        output_path = Path(
            self.output_dir,
            country,
            city,
            station_id,
            variable,
            f"cams_{variable}_{country}_{city}_{station_id}_{time_range}{ext}"
        )
        return output_path

    def save_data(self, data, output_path):
        pass
