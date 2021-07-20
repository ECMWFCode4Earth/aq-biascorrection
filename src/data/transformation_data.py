import concurrent.futures
import logging
import os
import pathlib
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.constants import ROOT_DIR
from src.data.transformation_location import LocationTransformer
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
