import glob
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from ipywidgets import interactive, widgets
from matplotlib.dates import DateFormatter
from pydantic.dataclasses import dataclass

from src.constants import ROOT_DIR
from src.data.utils import Location
from src.logging import get_logger
from src.metrics.validation_metrics import ValidationTables
from src.visualization.validation_visualization import ValidationVisualization

df_stations = pd.read_csv(ROOT_DIR / "data" / "external" / "stations.csv", index_col=0)
date_form = DateFormatter("%-d %b %y")

logger = get_logger('Model Predictions Validation')


class ValidationDataset:
    def __init__(
            self,
            cams: pd.DataFrame,
            observations: pd.DataFrame,
            predictions: pd.DataFrame,
            class_on_train: str
    ):
        self.cams = cams
        self.observations = observations
        self.predictions = predictions
        self.class_on_train = class_on_train


@dataclass
class Validator:
    model_name: str
    varname: str
    visualizations_output_dir: Path
    metrics_output_dir: Path

    def __post_init__(self):
        self.visualizations_output_dir = self.visualizations_output_dir / \
                                         self.varname / \
                                         'results'
        if not self.visualizations_output_dir.exists():
            os.makedirs(self.visualizations_output_dir, exist_ok=True)

        self.metrics_output_dir = self.metrics_output_dir / \
                                  self.varname / \
                                  'results'
        if not self.metrics_output_dir.exists():
            os.makedirs(self.metrics_output_dir, exist_ok=True)

    def run(
            self,
            station_code: str,
            class_on_train: str = 'all'):
        logger.info(f'Starting Validation worfklow for variable '
                    f'{self.varname} and station {station_code}.')
        logger.info('Getting the data from the machine learning predictions.')
        ml_predictions = self.load_model_predictions(
            station_code,
            class_on_train
        )
        logger.info('Getting the data from the CAMS Forecast and Observations.')
        cams_and_obs = self.load_obs_and_cams(
            station_code
        )
        logger.info('Creating a ValidationDataset object for every run.')
        validation_datasets = self.get_initialization_datasets(
            ml_predictions,
            cams_and_obs
        )
        logger.info('Running ValidationVisualization workflow.')
        ValidationVisualization(
            validation_datasets,
            self.varname,
            Location.get_location_by_id(station_code),
            class_on_train,
            self.visualizations_output_dir
        ).run()
        logger.info('Running ValidationTables workflow')
        ValidationTables(
            validation_datasets,
            Location.get_location_by_id(station_code),
            self.metrics_output_dir
        ).run()

    def load_model_predictions(
            self,
            station: str,
            data_type: str
    ) -> pd.DataFrame:
        """
        Method to load the machine learning model predictions for a given station.
        The predictions collected are differentiated between "train" and "test".
        Args:
            station: id of the stations from which take the predictions
            data_type: "train", "test" or "all" refers to the data that is taken
        """
        directory = ROOT_DIR / "models" / "results" / self.model_name / self.varname
        sum_dfs = {"train": None,
                   "test": None}
        for key in sum_dfs.keys():
            sum_df, count = None, 0
            for file in glob.glob(str(directory / f"*_{key}.csv")):
                count += 1
                df = pd.read_csv(file, index_col=[0, 1])
                if sum_df is not None:
                    sum_df += df
                else:
                    sum_df = df
            sum_df /= count
            sum_df['class_on_train'] = key
            sum_dfs[key] = sum_df.loc[(slice(None), station), :].droplevel(1)
        if data_type == 'all':
            df_total = pd.concat([sum_dfs['train'], sum_dfs['test']])
        elif data_type == 'train':
            df_total = sum_dfs['train']
        elif data_type == 'test':
            df_total = sum_dfs['train']
        else:
            raise NotImplementedError(
                'The data type has to be equal to: "train", "test" or "all"'
            )
        df_total = df_total.sort_index()
        return df_total

    def load_obs_and_cams(
            self,
            station: str
    ) -> pd.DataFrame:
        """
        Method to load the CAMS forecast predictions and OpenAQ observations
        for a given station.
        Args:
            station: id of the stations from which take the predictions
        """
        idir = ROOT_DIR / "data" / "processed"
        data_file = list(idir.rglob(f"data_{self.varname}_{station}.csv"))[0]
        data = pd.read_csv(data_file, index_col=0)
        data['index'] = pd.to_datetime(data['index'])
        return data.set_index('index')

    def get_initialization_datasets(
            self,
            df: pd.DataFrame,
            data: pd.DataFrame
    ) -> List[ValidationDataset]:
        """
        Method to transform the 24 machine learning predictions columns into a single
        column dataframe with the temporal data.
        Args:
            df: machine learning predictions for correcting the CAMS forecast
            data: CAMS forecast data and observations
        """
        init_datasets = []
        for init_time, values in df.iterrows():
            indices = pd.date_range(
                start=init_time,
                periods=len(values) - 1,
                freq='H'
            )
            # Perform the correction of the forecasts
            predictions = data.loc[
                              indices, f"{self.varname}_forecast"
                          ] - values[:-1].values
            predictions = predictions.to_frame(
                'CAMS + Correction'
            ).astype(float)
            cams = data[f"{self.varname}_forecast"].loc[predictions.index].to_frame(
                'CAMS Forecast'
            ).astype(float)
            obs = data[f"{self.varname}_observed"].loc[predictions.index].to_frame(
                'Observations'
            ).astype(float)
            class_on_train = values[-1]
            init_datasets.append(
                ValidationDataset(cams, obs, predictions, class_on_train)
            )
        return init_datasets


# Methods for implementation of Jupyter Tool
def get_all_locations() -> List[str]:
    return list(df_stations.city.unique())


def get_id_location(city: str) -> str:
    return df_stations.loc[df_stations.city == city, "id"].values[0]


def interactive_viz(varname: str, station: str, date_range: tuple):
    plotter = ValidationVisualization("InceptionTime_ensemble", varname)
    plotter.run(get_id_location(station), date_range)


if __name__ == '__main__':
    stations = pd.read_csv(f"{ROOT_DIR}/data/external/stations.csv")
    for station_id in stations['id'].values:
        try:
            Validator(
                'InceptionTime_ensemble',
                'no2',
                ROOT_DIR / 'reports' / 'figures',
                ROOT_DIR / 'reports' / 'tables'
            ).run(station_id)
        except:
            pass
