import concurrent.futures
import datetime
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import xarray as xr
from dask import distributed

from src.data import utils
from src.data.utils import Location


logger = logging.getLogger("CAMS Processor")


class CAMSProcessor:
    """
    Class to process the CAMS model forecast
    """

    def __init__(
            self,
            input_dir: Path,
            intermediary_dir: Path,
            locations_csv: Path,
            output_dir: Path,
            time_range: Dict[str, str] = None
    ):
        self.input_dir = input_dir
        self.intermediary_dir = intermediary_dir
        if time_range is None:
            self.time_range = dict(start='2019-06-01', end='2021-03-31')
        else:
            self.time_range = time_range

        self.locations_df = pd.read_csv(locations_csv)
        self.output_dir = output_dir

    def run(self) -> str:
        """
        Main method to run the CAMSProcessor steps
        """
        initialization_times = self.get_initialization_times()
        total_data, intermediary_paths = self.get_total_data(
            initialization_times
        )
        # We get the data for all initialization_times and locations
        for i, location in enumerate(self.locations_df.iterrows()):
            # Write one netcdf for each location of interest
            loc = Location(
                location[1]['id'],
                location[1]['city'],
                location[1]['country'],
                location[1]['latitude'],
                location[1]['longitude'],
                location[1]['timezone'],
                location[1]['elevation']
            )
            output_path_location = loc.get_forecast_path(
                self.output_dir,
                '_'.join(
                    self.time_range.values()
                ).replace('-', '')
            )
            data_location = total_data.sel(station_id=loc.location_id)
            logger.info(f'Writing netcdf for location {i} out of '
                        f'{len(self.locations_df)} with id: '
                        f'{location[1]["id"]}')
            utils.write_netcdf(output_path_location, data_location)
        logger.info(f'Deleting intermediary data')
        utils.remove_intermediary_paths(intermediary_paths)
        return 'Data has been processed successfully'

    def get_initialization_times(self) -> List[str]:
        """
        Get all the initialization times (days) in the range defined by the
        time_range argument of the CAMSPreprocessor class
        """
        initialization_times = pd.date_range(
            self.time_range['start'],
            self.time_range['end'],
            freq='1D'
        )
        initialization_times = [datetime.datetime.strftime(
            time, '%Y%m%d'
        ) for time in initialization_times]
        return initialization_times

    def get_total_data(
            self,
            initialization_times: List[str]) -> Tuple[xr.Dataset, Path]:
        """
        Get the data for the whole time_range defined as an argument and all
        the locations of interest given in the .csv file concatenated
        """
        intermediary_paths = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_entry = {
                executor.submit(
                    self.get_data_for_initialization_time,
                    init_time
                ): init_time for init_time in initialization_times}
            for future in concurrent.futures.as_completed(future_to_entry):
                intermediary_path_for_init_time = future.result()
                intermediary_paths.append(intermediary_path_for_init_time)
        logger.info(f'Opening the data for all the stations and times')
        total_data = xr.open_mfdataset(intermediary_paths,
                                       concat_dim='time')
        return total_data, intermediary_paths

    def get_data_for_initialization_time(self,
                                         initialization_time: str) -> Path:
        """
        Get the data for one initialization_time of all the different locations
        of interest given in the .csv file
        """
        logger.info(f'Getting data for initialization time'
                     f' {initialization_time}')
        try:
            intermediary_path = self.get_intermediary_path(initialization_time)
            if not intermediary_path.exists():
                paths_for_forecast = self.get_paths_for_forecasted_variables(
                    initialization_time
                )
                data = self.get_data(paths_for_forecast)
                utils.write_netcdf(intermediary_path, data)
            return intermediary_path
        except Exception as ex:
            logger.error(ex)

    def get_paths_for_forecasted_variables(
            self,
            initialization_time: str
    ) -> List[Path]:
        """
        Get all the paths associated with an initialization_time sorted by name
        """
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

    def get_data(self, paths_for_forecast) -> xr.Dataset:
        """
        Get the data of the CAMS model for every location defined in the
        .csv which gathers the locations of interest
        """
        data = xr.open_mfdataset(paths_for_forecast,
                                 concat_dim='time',
                                 preprocess=self.filter_location)
        return data

    def filter_location(self, data: xr.Dataset) -> xr.Dataset:
        """
        Filter method to use in xr.open_mfdataset
        """
        data_for_stations = []
        for location in self.locations_df.iterrows():
            loc = Location(
                location[1]['id'],
                location[1]['city'],
                location[1]['country'],
                location[1]['latitude'],
                location[1]['longitude'],
                location[1]['timezone'],
                location[1]['elevation']
            )
            data_location = data.sel(latitude=loc.latitude,
                                     longitude=loc.longitude,
                                     method='nearest')
            data_location['station_id'] = loc.location_id
            data_location = data_location.set_coords('station_id')
            data_location = data_location.expand_dims('station_id')
            data_for_stations.append(data_location)
        data = xr.concat(data_for_stations, dim='station_id')
        return data

    def get_intermediary_path(self, initialization_time):
        ext = '.nc'
        intermediary_path = Path(
            self.intermediary_dir,
            f"cams_all_stations_{initialization_time}{ext}"
        )
        return intermediary_path


