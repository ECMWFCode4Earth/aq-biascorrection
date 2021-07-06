import glob
from pathlib import Path
from src.constants import ROOT_DIR

import pandas as pd
from datacleaner import autoclean
from sklearn.model_selection import train_test_split
from pydantic.dataclasses import dataclass


@dataclass
class DataLoader:
    variable: str
    input_dir: Path = ROOT_DIR / "data" / "processed"

    def data_load(self):
        # Get the data for all the stations available for the given variable
        data = self.get_data_for_all_stations_available()
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
        return {'train': (X_train, y_train), 'test': (X_test, y_test)}

    def get_data_for_all_stations_available(self):
        files = glob.glob(f"{self.input_dir}/{self.variable}/*.csv")
        data = None
        for file in files:
            dataset = pd.read_csv(file, index_col=0)
            data = data.append(dataset) if data is not None else dataset
        return data

