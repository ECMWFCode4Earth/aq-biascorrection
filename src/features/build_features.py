from src.data.utils import Location

import logging
import pandas as pd
import numpy as np
from typing import Tuple
from pydantic.dataclasses import dataclass
from joblib import Memory

from src.logging import get_logger

logger = get_logger("Feature Builder")

mem = Memory(cachedir='/tmp', verbose=1)

@dataclass
class FeatureBuilder:
    """
    Class that generates the dataset corresponding to a station that can be used
    for model training and inference.

    Attributes:
        n_prev_obs (int): Number of previous forecast and errors to consider.
        n_future (int): Number of following bias to predict.
        min_st_obs (int): Minimum number of observations required at one station to be
        considered.
    """

    n_prev_obs: int
    n_future: int
    min_st_obs: int = None

    def __post_init__(self):
        if self.min_st_obs is None:
            self.min_st_obs = self.n_future + self.n_prev_obs
        self.build = mem.cache(self.build)

    def build(
        self,
        filename: str,
        include_time_attrs: bool = True,
        categorical_to_numeric: bool = True,
        include_station_attrs: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates features and labels dataset. The columns are labeled using the
        following guideline:
            - { variable }_{ type }_{ freq }
        where the variable represents the air quality variable, the type represents
        whether it corresponds to a forecast or a observation, and the freq represents
        the previous time (if it is a number). If freq = 'attr' then the column
        represents a variable that is related to the station so it does not change. If
        freq = 'aux', the variable corresponds to extra information that it is not an
        air quality variable.

        Args:
            filename (str): name of the file containing the data for the station.
            include_time_attrs (bool): whether to include the hour and the month as
            features.
            categorical_to_numeric (bool): whether to transform categorical variables
            (month or hour) to numeric.
            include_station_attrs (bool): whether to include the station attributes like
            longitude, latitude and altitude as features.

        Returns:
            pd.DataFrame: features to feed a model for the given station.
            pd.DataFrame: values to predict for the given station.
        """
        logger.info(f"Reading data from {filename}")
        dataset = pd.read_csv(filename, index_col=1, parse_dates=True)
        var, st_code = filename.replace(".csv", "").split("_")[-2:]
        loc = Location.get_location_by_id(st_code)
        aux = self.get_features_hour_and_month(
            dataset[["local_time_hour"]], categorical_to_numeric
        )
        dataset = dataset.drop(
            ["Unnamed: 0", "local_time_hour"], axis=1, errors="ignore"
        )

        # Skip if there is no the minimum number of observations required.
        if len(dataset.index) < self.min_st_obs:
            return pd.DataFrame(), pd.DataFrame()

        # Get features and labels
        samples = self.get_samples(dataset)
        if len(samples) == 0:
            return pd.DataFrame(), pd.DataFrame()
        else:
            data_samples = pd.concat(samples)

        if include_time_attrs:
            data_samples = data_samples.merge(aux, left_index=True, right_index=True)
        if include_station_attrs:
            data_samples["latitude_attr"] = loc.latitude
            data_samples["longitude_attr"] = loc.longitude
            data_samples["elevation_attr"] = loc.elevation
        X, y = self.get_features_and_labels(data_samples)

        index = set(X.index.values).intersection(y.index.values)
        return X.loc[index, :], y.loc[index, :]

    def get_features_and_labels(
            self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        columns_to_features = []
        columns_to_labels = []
        for column in dataset.columns:
            if "bias" in column:
                columns_to_features.append(column)
            else:
                columns_to_labels.append(column)
        X = dataset.drop(columns=columns_to_features)
        y = dataset.drop(columns=columns_to_labels)
        return X, y


    def get_samples(self, dataset: pd.DataFrame) -> pd.DataFrame:
        number_per_sample = self.n_future + self.n_prev_obs
        samples = []
        i = 0
        while i < len(dataset) - number_per_sample:
            sample = dataset.iloc[i: i + number_per_sample]
            # Check that the first time of the sample is 00:00 or 12:00 utc
            if sample.index[number_per_sample - self.n_future].hour not in [0, 12]:
                i += 1
                continue
            # Check that all times are continuous (not more than 1h between times)
            if not np.all(np.diff(sample.index.values) == np.timedelta64(1, 'h')):
                i += 1
                continue
            data = {}
            for t, row in enumerate(sample.iterrows()):
                row = row[1]
                for feature in list(row.index):
                    data[f"{feature}_{t}"] = row[feature]
            i += 1
            samples.append(
                pd.DataFrame(
                    data,
                    index=[sample.index[number_per_sample - self.n_future]]
                )
            )
        return samples
    @staticmethod
    def get_features_hour_and_month(
        dataset: pd.DataFrame, categorical_to_numeric: bool = True
    ) -> pd.DataFrame:
        """
        Computes a dataframe with the sine and cosine decompositions of the month and
        local hour variables.
        This is made to represent the seasonality of the this features.

        Args:
            dataset (pd.DataFrame): Dataframe with a column named 'local_time_hour' and
            indexed by timestamps.

        Returns:
            pd.DataFrame: Table with 4 columns corresponding to the Cosine and Sine
                          decompositions of the month and hour variables.
        """
        df = pd.DataFrame(index=dataset.index)
        if categorical_to_numeric:
            df["hour_cos_aux"] = np.cos(dataset[["local_time_hour"]] * (2 * np.pi / 24))
            df["hour_sin_aux"] = np.sin(dataset[["local_time_hour"]] * (2 * np.pi / 24))
            df["month_cos_aux"] = np.cos(dataset.index.month * (2 * np.pi / 12))
            df["month_sin_aux"] = np.sin(dataset.index.month * (2 * np.pi / 12))
        else:
            df["hour"] = dataset[["local_time_hour"]]
            df["month"] = dataset.index.month
        return df
