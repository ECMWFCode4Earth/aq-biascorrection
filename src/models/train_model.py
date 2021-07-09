import os
import yaml
import logging
import warnings
from pathlib import Path
from typing import Dict, NoReturn

import xgboost as xg
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from src.features.load_dataset import DatasetLoader
from src.models.inception_time import InceptionTime
from src.models.regression import ElasticNetRegr
from src.models.utils import read_yaml
from src.constants import ROOT_DIR

warnings.filterwarnings('ignore')

logger = logging.getLogger("Model trainer")

models_dict = {
    'xgboost_regressor': xg.XGBRegressor, 
    'inception_time': InceptionTime,
    'elasticnet_regressor': ElasticNetRegr
}


class ModelTrain:
    """ Class that handles the model selection, training and validation of any set of
    models with a common structure (having methods: fit, model, predict, set_params, 
    get_params, save and load)
    
    Attributes:
        variable (str): Air quality variable to correct.
        idir (Path): Directory of the input data for the model.
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
        config_folder: str = ROOT_DIR / "models" / "configuration"
    ):
        config = read_yaml(config_folder / config_yml_filename)
        self.variable = config['data']['variable']
        self.idir = ROOT_DIR / config['data']['idir']
        self.n_prev_obs = config['data']['n_prev_obs']
        self.n_future = config['data']['n_future']
        self.min_st_obs = config['data']['min_station_observations']
        self.models = config['models']

        logger.info(f'Loading data for variable {self.variable}')
        ds_loader = DatasetLoader(self.variable,
                                  self.n_prev_obs,
                                  self.n_future,
                                  self.min_st_obs,
                                  input_dir=self.idir)
        self.X_train, self.y_train, self.X_test, self.y_test = ds_loader.load()

        # Shuffle train dataset.
        columns_X = len(self.X_train.columns)
        df = pd.concat([self.X_train, self.y_train], axis=1)
        df = df.sample(frac=1)
        self.X_train = df.iloc[:, :columns_X]
        self.y_train = df.iloc[:, columns_X:]

    def run(self):
        # Iterate over each model.
        for i, model in enumerate(self.models):
            self.update_model_output_dir(model['name'])
            logger.info(f'Training and validating model {i+1} '
                        f'out of {len(self.models)}')
            logger.info(f'Training model with method {model["name"]}')
 
            if model['model_selection']:
                self.selection_traininig_and_evalution(model)
            else:
                self.training_and_evaluation(model)

    def training_and_evaluation(self, model: Dict):
        mo = models_dict[model['type']](**model['model_parameters'])
        self.evaluate_model(mo)
        return mo

    def selection_traininig_and_evalution(self, model: Dict):
        training_params = model['training_method']

        gridsearch = GridSearchCV(
            models_dict[model['type']](),
            model['model_parameters'],
            cv=training_params['cv'],
            scoring=training_params['scoring'],
            n_jobs=training_params['n_jobs'],
            verbose=training_params['verbose'])

        gridsearch.fit(self.X_train, self.y_train)
        self.evaluate_model(gridsearch.best_estimator_)

        return gridsearch.best_estimator_
    
    def evaluate_model(self, model) -> NoReturn:
        """
        Evaluate the model performance of a model in both trainig and test dataset.
        """
        # Model retraining with all training dataset.
        model.fit(self.X_train, self.y_train)
        self.save_model_and_predictions(model)

        # Evaluating performance in test dataset.
        logger.info("Evaluating performance on test set.")
        labels = self.y_test
        preds = model.predict(self.X_test)

        exp_var, maxerr, mae, rmse, r2, r2time = get_metric_results(preds, labels)
        # self.save_r2_with_time_structure(r2time, False)

        logger.info("Evaluating performance on train set.")
        labels = self.y_train
        preds = model.predict(self.X_train)

        # Compute metrics
        tr_exp_var, tr_maxerr, tr_mae, tr_rmse, tr_r2, tr_r2time = get_metric_results(
            preds, labels
        )
        # self.save_r2_with_time_structure(tr_r2time, True)

        print(
            f"-----------------------------------------------\n"
            f"--------{self.model_name:^31}--------\n"
            f"-----------------------------------------------\n"
            f"Exp. Var (test): {tr_exp_var:.4f}({exp_var:.4f})\n"
            f"Max error (test): {tr_maxerr} ({maxerr})\n"
            f"MAE (test): {tr_mae:.4f} ({mae:.4f})\n"
            f"RMSE (test): {tr_rmse:.4f} ({rmse:.4f})\n"
            f"R2 (test): {tr_r2:.4f} ({r2:.4f})\n")

        cams_max, cams_mae, cams_rmse = self.show_predictions_result()

        data = {
            'model': self.model_name,
            'variable': self.variable,
            'params': model.get_params(),
            'train': {
                'explained_variance': tr_exp_var,
                'max_errors': tr_maxerr,
                'mean_absolute_error': tr_mae,
                'root_mean_squared_error': tr_rmse,
                'r2': tr_r2
            }, 'test' : {
                'explained_variance': exp_var,
                'max_errors': maxerr,
                'mean_absolute_error': mae,
                'root_mean_squared_error': rmse,
                'r2': r2,
                'cams_max_err': cams_max,
                'cams_mae': cams_mae,
                'cams_rmse': cams_rmse
            }
        }

        # Save results.
        filename = f'allstations_{self.variable}_inception_time'
        logger.debug(f"Saving result of {self.model_name} to {self.results_output_dir}/"
                     f"test_{filename}.yml")
        with open(self.results_output_dir / f"test_{filename}.yml", 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def save_model_and_predictions(self, model) -> NoReturn:
        """ Save model fitted to the whole training dataset and its predictions for
        the test datasets considered.

        Args:
            model: model to save its weights, architecture and predictions of test set.        
        """
        # Save model 
        data_attrs = '_'.join([self.variable, str(self.n_prev_obs), str(self.n_future)])
        filename = f"{data_attrs}_{str(model)}"
        model.save(filename)
        # Save model predictions on test set.
        y_hat = model.predict(self.X_test, filename=filename)

    def save_r2_with_time_structure(self, r2_time, test: bool) -> NoReturn:
        outfile = self.results_output_dir / \
            f'plot_{"test" if test else "train"}_{self.variable}_" \
            f"r2_with_time_structure.png'
        logger.info(f"Plotting R2 with time structure to {outfile}")
        plt.figure(figsize=(12, 9))
        r2_time.plot(legend=False)
        plt.ylabel("R-Squared with time structure")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(outfile)

    def update_model_output_dir(self, model_name: str) -> NoReturn:
        self.model_name = model_name
        self.results_output_dir = ROOT_DIR / "models" / "results" / model_name
        os.makedirs(self.results_output_dir, exist_ok=True)

    def show_prediction_results(self):
        max_err = self.y_test.abs().max().round(4).values.tolist()
        mae = self.y_test.abs().mean()
        rmse = (self.y_test ** 2).mean() ** 0.5

        print(
            f"-----------------------------------------------\n"
            f"--------       CAMS predictions        --------\n"
            f"-----------------------------------------------\n"
            f"MAX ERR: {max_err}\n"
            f"MAE: {mae:.4f}\n"
            f"MSE: {mse:.4f}\n"
        )
        return max_err, mae, rmse

    @staticmethod
    def get_metric_results(preds: pd.DataFrame, labels: pd.DataFrame) -> tuple[float, ...]:
        """ Computes different metrics given the predictions and the true values.

        Args: 
            preds: predictions of any model.
            labels: true values of the predictions made.

        Returns:
            exp_var (float): the explained variance of the predictions.
            maxerr (float): the maximum error of the predictions.
            mae (float): the mean absolute error of the predictions.
            mse (float): the mean square error of the predictions.
            r2 (float): the R-squared value of the predictions.
            r2time (float): the R-squared with time structure value of the predictions. It
            only makes sense when the predictions correspond to a time series.
        """
        # Compute metrics
        exp_var = float(metrics.explained_variance_score(labels, preds))
        maxerr = (labels - preds).abs().max().round(4).values.tolist()
        mae = float(metrics.mean_absolute_error(labels, preds))
        rmse = float(metrics.mean_squared_error(labels, preds, squared=False))
        r2 = float(metrics.r2_score(labels, preds))    
        ssd = ((labels - preds) ** 2).cumsum()
        sst = (labels ** 2).cumsum()
        r2time = (sst - ssd) / sst.iloc[-1]
        return exp_var, maxerr, mae, rmse, r2, r2time
