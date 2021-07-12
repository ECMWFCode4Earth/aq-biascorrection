from src.data.utils import Location

import logging
import pandas as pd
import numpy as np
from pydantic.dataclasses import dataclass

logger = logging.getLogger("Feature Builder")


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

    def build(
            self,
            filename: str,
            include_time_attrs: bool = True,
            categorical_to_numeric: bool = True,
            include_station_attrs: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates features and labels dataset. The columns are labeled using the
        following guideline:
            - { variable }_{ type }_ { freq }
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
        var, st_code = filename.replace(".csv", "").split('_')[-2:]
        loc = Location.get_location_by_id(st_code)
        aux = self.get_features_hour_and_month(
            dataset[['local_time_hour']], categorical_to_numeric
        )
        dataset = dataset.drop(
            ['Unnamed: 0', 'local_time_hour'],
            axis=1, errors='ignore'
        )

        # Skip if there is no the minimum number of observations required.
        if len(dataset.index) < self.min_st_obs:
            return pd.DataFrame(), pd.DataFrame()

        # Get features and labels
        X = self.get_features(dataset.drop(f"{var}_bias", axis=1))
        if include_time_attrs:
            X = X.merge(aux, left_index=True, right_index=True)
        if include_station_attrs:
            X['latitude_attr'] = loc.latitude
            X['longitude_attr'] = loc.longitude
            X['elevation_attr'] = loc.elevation
        y = self.get_labels(dataset, f"{var}_bias")

        index = set(X.index.values).intersection(y.index.values)
        return X.loc[index, :], y.loc[index, :]

    def get_labels(self, dataset: pd.DataFrame, label: str) -> pd.DataFrame:
        obs_ahead = list(range(1, self.n_future + 1))
        ds = pd.DataFrame(columns=obs_ahead, index=dataset.index)
        for obs in obs_ahead:
            ds[obs] = dataset[label].shift(-obs)
        return ds.dropna()

    def get_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        past_obs = list(range(0, self.n_prev_obs))
        dfs = []
        for n_past in past_obs:
            dfs.append(dataset.shift(n_past))
        df = pd.concat(dfs, axis=1)
        index_past_val = np.array(past_obs * len(dataset.columns)).reshape(
            (-1, self.n_prev_obs)
        ).T.ravel()
        columns = list(map(lambda x: f"{x[0]}_{str(x[1])}",
                           zip(df.columns.values, index_past_val)))
        df.columns = columns
        return df.dropna()

    @staticmethod
    def get_features_hour_and_month(
        dataset: pd.DataFrame,
        categorical_to_numeric: bool = True
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
            df['hour_cos_aux'] = np.cos(dataset[['local_time_hour']] * (2 * np.pi / 24))
            df['hour_sin_aux'] = np.sin(dataset[['local_time_hour']] * (2 * np.pi / 24))
            df['month_cos_aux'] = np.cos(dataset.index.month * (2 * np.pi / 12))
            df['month_sin_aux'] = np.sin(dataset.index.month * (2 * np.pi / 12))
        else:
            df['hour'] = dataset[['local_time_hour']]
            df['month'] = dataset.index.month
        return df
