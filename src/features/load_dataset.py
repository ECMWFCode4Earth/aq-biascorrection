import glob
from pathlib import Path
from src.features.build_features import FeatureBuilder
from src.data.utils import Location
from src.constants import ROOT_DIR

import pandas as pd
import numpy as np
from datacleaner import autoclean
from sklearn.model_selection import train_test_split
from pydantic.dataclasses import dataclass


@dataclass
class DatasetLoader:
    """ Class to handle the generation of dataset for both training and testing.

    Attributes:
        variable (str): Air quality variable to consider. Choices are: pm25, o3 and no2.
        n_prev_obs (int): Number of previous times to consider at each time step.
        Default is 0, which means only predictions at current time are considered.
        n_futute (int): Number of future predictions to correct. Default is 1, which 
        means that only the next prediction is corrected.
        input_dir (Path): Directory to input data.
    """
    variable: str
    n_prev_obs: int = 0
    n_future: int = 1
    min_st_obs: int = 1
    input_dir: Path = ROOT_DIR / "data" / "processed"
    fb: FeatureBuilder = None

    def __post_init__(self):
        self.fb = FeatureBuilder(self.n_prev_obs, self.n_future, self.min_st_obs)

    def load(self) -> tuple[pd.DataFrame, ...]:
        # Get the data for all the stations available for the given variable
        data = self._join_each_station_dataset()
        # Shuffle the data
        data = data.sample(frac=1).reset_index(drop=True)
        data.drop(columns=['index'], inplace=True)
        data = autoclean(data)
        y_columns = [f'{self.variable}_bias', f'{self.variable}_observed']
        X = data.loc[:, ~data.columns.isin(y_columns)]
        for column in X.columns:
            X[column] = X[column].astype(float)
        y = data.loc[:, data.columns.isin([f'{self.variable}_bias'])]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )
        return X_train, y_train, X_test, y_test

    def _join_each_station_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        files = glob.glob(f"{self.input_dir}/{self.variable}/*.csv")
        train_data = None
        test_data = None
        
        # Iterate over all stations
        for station_file in files:
            train_ds, test_ds = self.fb.build(station_file)
            train_data = train_ds if train_data is None else train_data.append(train_ds)
            test_data = test_ds if test_data is None else test_data.append(test_ds)
        return train_data, test_data
