import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.data.utils import Location
from src.logging import get_logger

logger = get_logger("Validation Tables")


class ValidationTables:
    """
    Class to compute and save the tables with the scores:
    NMAE, BIAS, RMSE, PEARSON CORRELATION, Debiased-NMAE
    """

    def __init__(
        self,
        validation_datasets: list,
        location: Location,
        output_dir: Path,
    ):
        self.validation_datasets = validation_datasets
        self.location = location
        self.output_dir = output_dir

    def run(self):
        logger.info("Producing tables taking into account every single prediction")
        self.run_for_every_prediction()
        logger.info("Producing tables taking into account the whole time serie")
        self.run_for_the_complete_data()

    def run_for_every_prediction(self):
        metrics_train = []
        metrics_test = []
        for dataset in self.validation_datasets:
            cams = dataset.cams.values.flatten()
            observations = dataset.observations.values.flatten()
            predictions = dataset.predictions.values.flatten()
            persistence = dataset.persistence.values.flatten()
            data = self.metric_table(
                cams,
                observations,
                predictions,
                persistence,
                dataset.observations.index.values[0],
            )
            if dataset.class_on_train == "train":
                metrics_train.append(data)
            elif dataset.class_on_train == "test":
                metrics_test.append(data)
        data_train = pd.concat(metrics_train)
        data_train.loc[("Mean", "CAMS"), :] = data_train.xs(
            "CAMS", level=1, drop_level=False
        ).mean()
        data_train.loc[("Mean", "Corrected CAMS"), :] = data_train.xs(
            "Corrected CAMS", level=1, drop_level=False
        ).mean()
        data_train.loc[("Mean", "Persistence"), :] = data_train.xs(
            "Persistence", level=1, drop_level=False
        ).mean()
        data_test = pd.concat(metrics_test)
        data_test.loc[("Mean", "CAMS"), :] = data_test.xs(
            "CAMS", level=1, drop_level=False
        ).mean()
        data_test.loc[("Mean", "Corrected CAMS"), :] = data_test.xs(
            "Corrected CAMS", level=1, drop_level=False
        ).mean()
        data_test.loc[("Mean", "Persistence"), :] = data_test.xs(
            "Persistence", level=1, drop_level=False
        ).mean()
        dict_to_use_to_save = {"train": data_train, "test": data_test}
        for key, value in dict_to_use_to_save.items():
            city = "".join(self.location.city.split(" ")).lower()
            country = "".join(self.location.country.split(" ")).lower()
            station_code = "".join(self.location.location_id.split(" ")).lower()
            filename = (
                f"{key}_metrics-for-setofruns_" f"{station_code}_{city}_{country}.csv"
            )
            csv_path = self.output_dir / "SetOfRuns" / filename
            if not csv_path.parent.exists():
                os.makedirs(csv_path.parent, exist_ok=True)
            value.to_csv(csv_path)
        return data_train, data_test

    def run_for_the_complete_data(self):
        datasets_train = []
        datasets_test = []
        for dataset in self.validation_datasets:
            data = dataset.cams.join(
                [dataset.observations, dataset.predictions, dataset.persistence]
            )
            for column in data.columns:
                data[column] = data[column].astype(float)
            if dataset.class_on_train == "train":
                datasets_train.append(data)
            elif dataset.class_on_train == "test":
                datasets_test.append(data)
        data_train = pd.concat(datasets_train)
        data_test = pd.concat(datasets_test)
        resampling_options = [
            "H",
            "3H",
            "6H",
            "12H",
            "D",
            "3D",
            "7D",
            "15D",
            "MS",
            "QS",
            "No Resampling",
        ]
        data_dict = {"train": data_train, "test": data_test}
        results_dict = {"train": [], "test": []}
        for key, value in data_dict.items():
            for resampling_option in resampling_options:
                if resampling_option == "No Resampling":
                    data_resampled = value
                else:
                    data_resampled = value.resample(resampling_option).mean()
                if np.any(np.isnan(data_resampled)):
                    data_resampled.dropna(inplace=True)
                cams = data_resampled["CAMS"].values
                observations = data_resampled["Observations"].values
                predictions = data_resampled["Corrected CAMS"].values
                persistence = data_resampled["Persistence"].values
                data = self.metric_table(
                    cams, observations, predictions, persistence, resampling_option
                )
                results_dict[key].append(data)
        data_train = pd.concat(results_dict["train"])
        data_test = pd.concat(results_dict["test"])
        dict_to_use_to_save = {"train": data_train, "test": data_test}
        for key, value in dict_to_use_to_save.items():
            city = "".join(self.location.city.split(" ")).lower()
            country = "".join(self.location.country.split(" ")).lower()
            station_code = "".join(self.location.location_id.split(" ")).lower()
            filename = (
                f"{key}_metrics-for-completetimeserie_"
                f"{station_code}_{city}_{country}.csv"
            )
            csv_path = self.output_dir / "CompleteTimeSerie" / filename
            if not csv_path.parent.exists():
                os.makedirs(csv_path.parent, exist_ok=True)
            value.to_csv(csv_path)
        return data_train, data_test

    def nmae_value(self, diff: np.array, obs: np.array):
        abs_diff = np.abs(diff)
        numerator = abs_diff.mean()
        nmae = numerator / obs.mean()
        return nmae

    def bias_value(self, diff: xr.Dataset):
        bias = diff.mean()
        return bias

    def nmae_debiased_value(self, forecast: np.array, observation: np.array):
        diff = forecast - observation
        bias = diff.mean()
        debiased_forecast = forecast - bias
        diff = debiased_forecast - observation
        return self.nmae_value(diff, observation)

    def rmse_value(self, diff: np.array):
        rmse = np.sqrt((diff ** 2).mean())
        return rmse

    def pearson_value(self, forecast: np.array, observation: np.array):
        pearson_correlation = np.corrcoef(forecast, observation)[0][1]
        return pearson_correlation

    def metric_table(self, cams, observations, predictions, persistence, index):
        tables_cams = {
            "NMAE": self.nmae_value(cams - observations, observations),
            "BIAS": self.bias_value(cams - observations),
            "RMSE": self.rmse_value(cams - observations),
            "De-Biased NMAE": self.nmae_debiased_value(cams, observations),
            "Pearson Correlation": self.pearson_value(cams, observations),
        }
        tables_predictions = {
            "NMAE": self.nmae_value(predictions - observations, observations),
            "BIAS": self.bias_value(predictions - observations),
            "RMSE": self.rmse_value(predictions - observations),
            "De-Biased NMAE": self.nmae_debiased_value(predictions, observations),
            "Pearson Correlation": self.pearson_value(predictions, observations),
        }
        tables_persistence = {
            "NMAE": self.nmae_value(persistence - observations, observations),
            "BIAS": self.bias_value(persistence - observations),
            "RMSE": self.rmse_value(persistence - observations),
            "De-Biased NMAE": self.nmae_debiased_value(persistence, observations),
            "Pearson Correlation": self.pearson_value(persistence, observations),
        }
        df_cams = pd.DataFrame(tables_cams, index=[index])
        df_cams["type"] = "CAMS"
        df_cams.set_index([df_cams.index, "type"], inplace=True)
        df_predictions = pd.DataFrame(tables_predictions, index=[index])
        df_predictions["type"] = "Corrected CAMS"
        df_predictions.set_index([df_predictions.index, "type"], inplace=True)
        df_persistence = pd.DataFrame(tables_persistence, index=[index])
        df_persistence["type"] = "Persistence"
        df_persistence.set_index([df_persistence.index, "type"], inplace=True)
        data = df_cams.append(df_predictions).append(df_persistence)
        return data
