import concurrent.futures
import glob
from pathlib import Path
from typing import Tuple

import pandas as pd
from joblib import Memory
from pydantic.dataclasses import dataclass

from src.constants import ROOT_DIR
from src.features.build_features import FeatureBuilder

memory = Memory(cachedir="/tmp", verbose=0)

from src.logger import get_logger

logger = get_logger("Dataset Loader")


class DatasetLoader:
    """
    Class to handle the generation of dataset for both training and testing.

    Attributes:
        variable (str): Air quality variable to consider. Choices are: pm25, o3 and no2.
        n_prev_obs (int): Number of previous times to consider at each time step.
        Default is 0, which means only predictions at current time are considered.
        n_futute (int): Number of future predictions to correct. Default is 1, which
        means that only the next prediction is corrected.
        input_dir (Path): Directory to input data.
    """

    def __init__(
        self,
        variable: str,
        n_prev_obs: int = 0,
        n_future: int = 1,
        min_st_obs: int = 1,
        input_dir: Path = ROOT_DIR / "data" / "processed",
        cached: bool = True,
    ):
        self.variable = variable
        self.n_prev_obs = n_prev_obs
        self.n_future = n_future
        self.min_st_obs = min_st_obs
        self.input_dir = input_dir
        self.fb = FeatureBuilder(self.n_prev_obs, self.n_future, self.min_st_obs)
        if cached:
            self.load = memory.cache(self.load)
            self.load_station = memory.cache(self.load_station)

    def load(
        self, split_ratio: float = 0.8, categorical_to_numeric: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Get the data for all the stations available for the given variable
        files = glob.glob(f"{self.input_dir}/{self.variable}/*.csv")
        X_train, y_train, X_test, y_test = None, None, None, None

        # Iterate over all stations
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_entry = {
                executor.submit(
                    self.load_station, station_file, split_ratio, categorical_to_numeric
                ): station_file
                for station_file in files
            }
            for future in concurrent.futures.as_completed(future_to_entry):
                training_sets = future.result()
                if X_train is None:
                    X_train = training_sets[0]
                    X_test = training_sets[1]
                    y_train = training_sets[2]
                    y_test = training_sets[3]
                else:
                    X_train = X_train.append(training_sets[0])
                    X_test = X_test.append(training_sets[1])
                    y_train = y_train.append(training_sets[2])
                    y_test = y_test.append(training_sets[3])
        return X_train, y_train, X_test, y_test

    def load_station(self, station_file, split_ratio, categorical_to_numeric):
        X, y = self.fb.build(
            station_file, categorical_to_numeric=categorical_to_numeric
        )
        if X is None:
            return None
        threshold = int(len(X.index) * split_ratio)
        X_train, y_train = X.iloc[:threshold, :], y.iloc[:threshold, :]
        X_test, y_test = X.iloc[threshold:, :], y.iloc[threshold:, :]
        return X_train, X_test, y_train, y_test

    def load_to_predict(
        self, split_ratio: float = 1.0, categorical_to_numeric: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Get the data for all the stations available for the given variable
        files = glob.glob(f"{self.input_dir}/{self.variable}/*.csv")
        X_train, y_train, X_test, y_test = None, None, None, None

        # Iterate over all stations
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_entry = {
                executor.submit(
                    self.load_station, station_file, split_ratio, categorical_to_numeric
                ): station_file
                for station_file in files
            }
            for future in concurrent.futures.as_completed(future_to_entry):
                training_sets = future.result()
                if X_train is None:
                    X_train = training_sets[0]
                    X_test = training_sets[1]
                    y_train = training_sets[2]
                    y_test = training_sets[3]
                else:
                    X_train = X_train.append(training_sets[0])
                    X_test = X_test.append(training_sets[1])
                    y_train = y_train.append(training_sets[2])
                    y_test = y_test.append(training_sets[3])
        return X_train, y_train, X_test, y_test
