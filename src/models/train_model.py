from pathlib import Path
from src.models.load_data import DataLoader
from src.data.utils import Location

import pandas as pd

import logging

class ModelTrainer:
    def __init__(
            self,
            variable: str,
            locations_csv_path: Path = Path('../../data/external/stations.csv')
    ):
        self.variable = variable
        self.locations = pd.read_csv(locations_csv_path)

    def run(self):
        data = self.data_load()

    def data_load(self):
        data_for_locations = None
        for location in self.locations.iterrow():
            loc = Location(
                location[1]['id'],
                location[1]['city'],
                location[1]['country'],
                location[1]['latitude'],
                location[1]['longitude'],
                location[1]['timezone']
            )
            try:
                data_for_location = DataLoader(self.variable, loc).run()
                if data_for_locations is None:
                    data_for_locations = data_for_location
                else:
                    data_for_locations = pd.concat([data_for_locations,
                                                    data_for_location])
            except Exception as ex:
                logging.info(ex)
                pass
        return data_for_locations

