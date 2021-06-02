from pathlib import Path
from typing import List, Dict
from src.data.transformation.transformation_location import LocationTransformer
from src.data.utils import Location

import pandas as pd

import concurrent.futures
import pathlib
import logging
import os

class DataTransformer:
    def __init__(
            self,
            variable: str,
            locations_csv_path: Path = Path(
                './data/external/stations_with_altitude.csv'
            ),
            output_dir: Path = Path('./data/processed/'),
            observations_dir: Path = Path('./data/interim/observations/'),
            forecast_dir: Path = Path('./data/interim/forecasts/'),
            time_range: Dict[str, str] = None
    ):
        self.variable = variable
        self.locations = pd.read_csv(locations_csv_path)
        self.output_dir = output_dir
        self.observations_dir = observations_dir
        self.forecast_dir = forecast_dir
        if time_range is None:
            time_range = dict(start='2019-06-01', end='2021-03-31')
        self.time_range = time_range

    def run(self):
        paths = self.data_transform()
        return paths

    def data_transform(self) -> List[Path]:
        data_for_locations_paths = []
        locations = [Location(
                location[1]['id'],
                location[1]['city'],
                location[1]['country'],
                location[1]['latitude'],
                location[1]['longitude'],
                location[1]['timezone'],
                location[1]['elevation']
            ) for location in self.locations.iterrows()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_entry = {
                executor.submit(
                    self._data_transform,
                    location
                ): location for location in locations}
            for future in concurrent.futures.as_completed(future_to_entry):
                result = future.result()
                if type(result) is pathlib.PosixPath:
                    logging.info(f'Intermediary data saved to:'
                                 f' {result}')
                    data_for_locations_paths.append(result)
                else:
                    logging.error(result)
        return data_for_locations_paths

    def _data_transform(self, loc) -> Path or Exception:
            try:
                logging.info(f'Extracting data for location: {str(loc)}')
                inter_loc_path = self.get_output_path(loc)
                if inter_loc_path.exists():
                    return inter_loc_path
                data_for_location = LocationTransformer(
                    self.variable,
                    loc,
                    observations_dir=self.observations_dir,
                    forecast_dir=self.forecast_dir,
                    time_range=self.time_range
                ).run()
                data_for_location.to_csv(str(inter_loc_path))
                return inter_loc_path
            except Exception as ex:
                return ex

    def get_output_path(self, loc: Location) -> Path:
        ext = '.csv'
        intermediary_path = Path(
            self.output_dir,
            self.variable,
            f"data_{self.variable}_{loc.location_id}{ext}"
        )
        if not intermediary_path.parent.exists():
            os.makedirs(intermediary_path.parent, exist_ok=True)
        return intermediary_path


if __name__ == '__main__':
    DataTransformer(
        variable="pm25",
        locations_csv_path=Path("/home/pereza/git/esowc/aq-biascorrection/data/external/stations_with_altitude.csv"),
        output_dir=Path("/home/pereza/git/esowc/aq-biascorrection/data/processed/"),
        observations_dir=Path("/home/pereza/git/esowc/aq-biascorrection/data/interim/observations/"),
        forecast_dir=Path("/home/pereza/git/esowc/aq-biascorrection/data/interim/forecasts/"),
        time_range=None
    ).run()