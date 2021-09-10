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

logger = get_logger("Model trainer")

models_dict = {
    "gradient_boosting": GradientBoosting,
    "inception_time": InceptionTime,
    "elasticnet_regressor": ElasticNetRegr,
}


class ModelTrain:
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
        config_folder: Path = ROOT_DIR / "models" / "configuration",
    ):
        config = read_yaml(config_folder / config_yml_filename)
        self.variable = config["data"]["variable"]
        self.input_dir = ROOT_DIR / config["data"]["idir"]
        self.n_prev_obs = config["data"]["n_prev_obs"]
        self.n_future = config["data"]["n_future"]
        self.min_st_obs = config["data"]["min_station_observations"]
        self.models = config["models"]
        self.categorical_to_numeric = True

        logger.info(f"Loading data for variable {self.variable}")
        self.ds_loader = DatasetLoader(
            self.variable,
            self.n_prev_obs,
            self.n_future,
            self.min_st_obs,
            input_dir=self.input_dir,
        )
        self.__build_datasets()

    def __build_datasets(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.ds_loader.load(
            categorical_to_numeric=self.categorical_to_numeric
        )

        # Shuffle train dataset.
        columns_X = len(self.X_train.columns)
        df = pd.concat([self.X_train, self.y_train], axis=1)
        df = df.sample(frac=1)
        df = df.set_index([df.index, df.station]).drop("station", axis=1)
        self.X_train = df.iloc[:, : (columns_X - 1)]
        self.y_train = df.iloc[:, (columns_X - 1) :]

        columns_X = len(self.X_test.columns)
        df_test = pd.concat([self.X_test, self.y_test], axis=1)
        df_test = df_test.set_index([df_test.index, df_test.station]).drop(
            "station", axis=1
        )
        self.X_test = df_test.iloc[:, : (columns_X - 1)]
        self.y_test = df_test.iloc[:, (columns_X - 1) :]

    def run(self):
        # Iterate over each model.
        for i, model in enumerate(self.models):
            self.update_model_output_dir(model["name"])
            self.update_datasets(model)
            logger.info(
                f"Training and validating model {i+1} " f"out of {len(self.models)}"
            )
            logger.info(f'Training model with method {model["name"]}')

            if model["model_selection"]:
                self.selection_train_and_evaluation(model)
            else:
                self.train_and_evaluation(model)

    @retry()
    def train_and_evaluation(self, model: Dict):
        for ensemble_number in range(model["model_ensemble"]):
            model["model_parameters"]["output_dims"] = self.n_future
            mo = models_dict[model["type"]](**model["model_parameters"])
            model_path, scaler_paths = self.train_model(mo, ensemble_number)
            try:
                self.evaluate_model(mo, ensemble_number)
            except Exception as ex:
                logger.info(ex)
                os.remove(model_path)
                for scaler_path in scaler_paths.values():
                    os.remove(scaler_path)
                raise Exception(ex)
        return mo

    def selection_train_and_evaluation(self, model: Dict):
        training_params = model["training_method"]

        gridsearch = GridSearchCV(
            models_dict[model["type"]](),
            model["model_parameters"],
            cv=training_params["cv"],
            scoring=training_params["scoring"],
            n_jobs=training_params["n_jobs"],
            verbose=training_params["verbose"],
        )

        gridsearch.fit(self.X_train, self.y_train)
        self.evaluate_model(gridsearch.best_estimator_)

        return gridsearch.best_estimator_

    def train_model(self, model, ensemble_number: int) -> NoReturn:
        """
        Train the model using the training dataset. If the model is stored in .h5
        format, it only loads the model without training it.
        """
        model_path, scaler_paths = self.get_model_and_scaler_output_path(
            model, ensemble_number
        )
        if (
            model_path.exists()
            and scaler_paths["attr_scaler"].exists()
            and scaler_paths["aq_vars_scaler"].exists()
        ):
            logger.info("Model and data scalers already exist, loading!")
            model.load(model_path, scaler_paths)
        else:
            logger.info("Model does not exist yet, training and saving!")
            try:
                model.fit(self.X_train, self.y_train)
            except Exception as ex:
                pass
            model.save(model_path, scaler_paths)
        return model_path, scaler_paths

    def evaluate_model(self, model, ensemble_number: int) -> NoReturn:
        """
        Evaluate the model performance of a model in both training and test dataset.
        """
        logger.info("Evaluating performance on test set.")
        labels = self.y_test
        predictions_output_path_test = self.get_model_predictions_path(
            model, ensemble_number, "test"
        )
        preds = model.predict(self.X_test, filepath=predictions_output_path_test)
        test_metrics = self.get_metric_results(preds, labels)
        te_exp_var, te_maxerr, te_mae, te_rmse, te_r2 = test_metrics
        # self.save_r2_with_time_structure(r2time, False)

        logger.info("Evaluating performance on train set.")
        labels = self.y_train
        predictions_output_path_train = self.get_model_predictions_path(
            model, ensemble_number, "train"
        )
        preds = model.predict(self.X_train, filepath=predictions_output_path_train)
        training_metrics = self.get_metric_results(preds, labels)
        tr_exp_var, tr_maxerr, tr_mae, tr_rmse, tr_r2 = training_metrics
        # self.save_r2_with_time_structure(tr_r2time, True)

        print(
            f"-----------------------------------------------\n"
            f"--------{self.model_name:^31}--------\n"
            f"-----------------------------------------------\n"
            f"Exp. Var (test): {tr_exp_var:.4f}({te_exp_var:.4f})\n"
            f"Max error (test): {tr_maxerr} ({te_maxerr})\n"
            f"MAE (test): {tr_mae:.4f} ({te_mae:.4f})\n"
            f"RMSE (test): {tr_rmse:.4f} ({te_rmse:.4f})\n"
            f"R2 (test): {tr_r2:.4f} ({te_r2:.4f})\n"
        )

        cams_max, cams_mae, cams_rmse = self.show_prediction_results()

        print(
            f"-----------------------------------------------\n"
            f"--------       CAMS predictions        --------\n"
            f"-----------------------------------------------\n"
            f"Max error: {cams_max}\n"
            f"MAE: {cams_mae}\n"
            f"RMSE: {cams_rmse}\n"
        )

        if 0.8 * cams_mae < tr_mae and 0.8 * cams_mae < te_mae:
            raise Exception("The model is not performing correctly")

        data = {
            "model": self.model_name,
            "variable": self.variable,
            "params": model.get_params(),
            "train": {
                "explained_variance": tr_exp_var,
                "max_error": tr_maxerr,
                "mean_absolute_error": tr_mae,
                "root_mean_squared_error": tr_rmse,
                "r2": tr_r2,
            },
            "test": {
                "explained_variance": te_exp_var,
                "max_error": te_maxerr,
                "mean_absolute_error": te_mae,
                "root_mean_squared_error": te_rmse,
                "r2": te_r2,
            },
            "cams_forecast": {
                "max_error": cams_max,
                "mean_absolute_error": cams_mae,
                "root_mean_squared_error": cams_rmse,
            },
        }
        # Save results.
        filename = f"allstations_{self.variable}_inception_time_{ensemble_number}"
        logger.debug(
            f"Saving result of {self.model_name} to {self.results_output_dir}/"
            f"test_{filename}.yml"
        )
        with open(self.results_output_dir / f"test_{filename}.yml", "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

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

    def get_model_predictions_path(
        self, model, ensemble_number: int, data_type: str
    ) -> Path:
        """
        Get the model predictions path

        Args:
            model: model that will be used for making the predictions
            ensemble_number: identifier for the ensemble member
            data_type: kind of data used to get the predictions (train or test)
        """
        data_attrs = "_".join([self.variable, str(self.n_prev_obs), str(self.n_future)])
        filename = f"{data_attrs}_{str(model)}_{ensemble_number}_{data_type}"
        predictions_path = self.results_output_dir / f"{filename}.csv"
        return predictions_path

    def save_r2_with_time_structure(self, r2_time, test: bool) -> NoReturn:
        outfile = (
            self.results_output_dir
            / f'plot_{"test" if test else "train"}_{self.variable}_" \
            f"r2_with_time_structure.png'
        )
        logger.info(f"Plotting R2 with time structure to {outfile}")
        plt.figure(figsize=(12, 9))
        r2_time.plot(legend=False)
        plt.ylabel("R-Squared with time structure")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(outfile)

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

    def update_datasets(self, model: Dict) -> NoReturn:
        if "categorical_to_numerical" in model.keys():
            new = model["categorical_to_numeric"]
        else:
            new = True  # default option

        if new != self.categorical_to_numeric:
            self.categorical_to_numeric = new
            self.__build_datasets()

    def show_prediction_results(self):
        max_err = round(float(self.y_test.abs().max().round(4).max()), 3)
        mae = round(float(self.y_test.abs().mean().round(4).mean()), 3)
        rmse = round(float(((self.y_test ** 2).mean() ** 0.5).round(4).mean()), 3)
        return max_err, mae, rmse

    @staticmethod
    def get_metric_results(
        preds: pd.DataFrame, labels: pd.DataFrame
    ) -> Tuple[float, ...]:
        """Computes different metrics given the predictions and the true values.

        Args:
            preds: predictions of any model.
            labels: true values of the predictions made.

        Returns:
            exp_var (float): the explained variance of the predictions.
            maxerr (float): the maximum error of the predictions.
            mae (float): the mean absolute error of the predictions.
            mse (float): the mean square error of the predictions.
            r2 (float): the R-squared value of the predictions.
        """
        # Compute metrics
        if sorted(labels.columns.values) != sorted(preds.columns.values):
            columns_rename_dict = {}
            for i in range(len(labels.columns.values)):
                columns_rename_dict[labels.columns[i]] = preds.columns[i]
            labels = labels.rename(columns=columns_rename_dict)
        exp_var = round(float(metrics.explained_variance_score(labels, preds)), 3)
        maxerr = round(float((labels - preds).abs().max().round(4).max()), 3)
        mae = round(float(metrics.mean_absolute_error(labels, preds)), 3)
        rmse = round(float(metrics.mean_squared_error(labels, preds, squared=False)), 3)
        r2 = round(float(metrics.r2_score(labels, preds)), 3)
        return exp_var, maxerr, mae, rmse, r2


if __name__ == "__main__":
    ModelTrain(
        "config_inceptiontime_depth6.yml"
    ).run()
