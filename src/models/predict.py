import os
import warnings
from pathlib import Path
from typing import Dict, NoReturn, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from tenacity import retry

from src.constants import ROOT_DIR
from src.features.load_dataset import DatasetLoader
from src.models.config.gradient_boosting import GradientBoosting
from src.models.config.inception_time import InceptionTime
from src.models.config.regression import ElasticNetRegr
from src.models.utils import read_yaml

warnings.filterwarnings("ignore")

from src.logger import get_logger

logger = get_logger("Model predict")

models_dict = {
    "gradient_boosting": GradientBoosting,
    "inception_time": InceptionTime,
    "elasticnet_regressor": ElasticNetRegr,
}


class ModelPredict:
    """Class that handles the model selection, training and validation of any set of
    models with a common structure (having methods: fit, model, predict, set_params,
    get_params, save and load)

    Attributes:
        variable (str): Air quality variable to correct.
        input_dir (Path): Directory of the input data for the model.
        results_output_dir (Path): Directory to the output data for the model.
        models (dict): collection of Models to train and validate.
        X_train (pd.DataFrame): features used for training purposes.
        y_train (pd.DataFrame): labels of the training instances.
        X_test (pd.DataFrame): features used for assesing the performance of the model.
        y_test (pd.DataFrame): labels of the test instances.
    """

    def __init__(
        self,
        config_yml_filename: str,
        input_data_dir: Path = ROOT_DIR / "data" / "processed",
        predictions_dir: Path = Path("/tmp"),
    ):
        config = read_yaml(config_yml_filename)
        self.variable = config["data"]["variable"]
        self.input_dir = input_data_dir
        self.n_prev_obs = config["data"]["n_prev_obs"]
        self.n_future = config["data"]["n_future"]
        self.min_st_obs = config["data"]["min_station_observations"]
        self.models = config["models"]
        self.categorical_to_numeric = True
        self.predictions_dir = predictions_dir

        logger.info(f"Loading data for variable {self.variable}")
        self.ds_loader = DatasetLoader(
            self.variable,
            self.n_prev_obs,
            self.n_future,
            self.min_st_obs,
            input_dir=self.input_dir,
            cached=False,
        )
        self.__build_datasets()

    def __build_datasets(self):
        self.X, self.y, self.X_test, self.y_test = self.ds_loader.load_to_predict(
            categorical_to_numeric=self.categorical_to_numeric
        )

        columns_X = len(self.X.columns)
        df = pd.concat([self.X, self.y], axis=1)
        df = df.sample(frac=1)
        df = df.set_index([df.index, df.station]).drop("station", axis=1)
        self.X = df.iloc[:, : (columns_X - 1)]

    def run(self):
        predictions_paths = []
        for i, model in enumerate(self.models):
            self.update_model_output_dir(model["name"])
            for ensemble_number in range(model["model_ensemble"]):
                model["model_parameters"]["output_dims"] = self.n_future
                mo = models_dict[model["type"]](**model["model_parameters"])
                model_path, scaler_paths = self.get_model_and_scaler_output_path(
                    mo, ensemble_number
                )
                mo.load(model_path, scaler_paths)
                predictions_output_path = (
                    self.predictions_dir / f"predictions_{ensemble_number}.csv"
                )
                preds = mo.predict(self.X, filepath=predictions_output_path)
                predictions_paths.append(predictions_output_path)
        return predictions_paths

    def update_model_output_dir(self, model_name: str) -> NoReturn:
        self.model_name = model_name
        self.results_output_dir = (
            ROOT_DIR / "models" / "results" / model_name / self.variable
        )
        self.weights_output_dir = (
            ROOT_DIR / "models" / "weights_storage" / model_name / self.variable
        )
        os.makedirs(self.results_output_dir, exist_ok=True)
        os.makedirs(self.weights_output_dir, exist_ok=True)

    def get_model_and_scaler_output_path(
        self, model, ensemble_number: int
    ) -> Tuple[Path, Dict]:
        """
        Get paths to save the model and the scalers once the model has been trained

        Args:
            model: model to save its weights and architecture.
            ensemble_number: identifier for the ensemble member
        """
        data_attrs = "_".join([self.variable, str(self.n_prev_obs), str(self.n_future)])
        filename = f"{data_attrs}_{str(model)}_{ensemble_number}"
        model_path = self.weights_output_dir / f"{filename}.h5"
        scaler_paths = {
            "attr_scaler": self.weights_output_dir / f"{filename}_attrscaler.pkl",
            "aq_vars_scaler": self.weights_output_dir / f"{filename}_aqvarsscaler.pkl",
            "aq_bias_scaler": self.weights_output_dir / f"{filename}_aqbiasscaler.pkl",
        }
        return model_path, scaler_paths
