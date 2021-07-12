import glob
from pathlib import Path
from src.features.build_features import FeatureBuilder
from src.constants import ROOT_DIR

import pandas as pd
import numpy as np
from datacleaner import autoclean
from pydantic.dataclasses import dataclass


@dataclass
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
    variable: str
    n_prev_obs: int = 0
    n_future: int = 1
    min_st_obs: int = 1
    input_dir: Path = ROOT_DIR / "data" / "processed"
    fb: FeatureBuilder = None

    def __post_init__(self):
        self.fb = FeatureBuilder(
            self.n_prev_obs,
            self.n_future,
            self.min_st_obs
        )

    def load(
        self,
        split_ratio: float = 0.8,
        categorical_to_numeric: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Get the data for all the stations available for the given variable
        files = glob.glob(f"{self.input_dir}/{self.variable}/*.csv")
        X_train, y_train, X_test, y_test = None, None, None, None

        # Iterate over all stations
        for station_file in files:
            X, y = self.fb.build(
                station_file, categorical_to_numeric=categorical_to_numeric)
            if X is None: continue  # Stations not satisfying min obs. requirement.
            threshold = int(len(X.index) * split_ratio)
            if X_train is None:
                X_train, y_train = X.iloc[:threshold, :], y.iloc[:threshold, :]
                X_test, y_test = X.iloc[threshold:, :], y.iloc[threshold:, :]
            else:
                X_train = X_train.append(X.iloc[:threshold, :])
                y_train = y_train.append(y.iloc[:threshold, :])
                X_test = X_test.append(X.iloc[threshold:, :])
                y_test = y_test.append(y.iloc[threshold:, :])
        return X_train, y_train, X_test, y_test
