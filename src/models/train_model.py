import os
import yaml
import logging
import warnings
from pathlib import Path
from typing import Dict, NoReturn

import xgboost as xg
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from src.data.load.load_data import DataLoader
from src.models.inception_time import InceptionTime
from src.models.utils import read_yaml
from src.constants import ROOT_DIR

warnings.filterwarnings('ignore')

logger = logging.getLogger("Model trainer")

models_dict = {
    'xgboost_regressor': xg.XGBRegressor, 
    'inception_time': InceptionTime
}


class ModelTrain:
    """ Class that handles the model selection, training and validation of any set of
    models with a common structure (having methods: fit, model, predict, set_params, 
    get_params, save and load)
    
    Attributes:
        variable (str): Air quality variable to correct.
        idir (str): Directory of the input data for the model.
        odir (str): Directory to the output data for the model.
        models (dict): collection of Models to train and validate.
        X_train (pd.DataFrame): features used for training purposes.
        y_train (pd.DataFrame): labels of the training instances.
        X_test (pd.DataFrame): features used for assesing the performance of the model.
        y_test (pd.DataFrame): labels of the test instances.
    """
    def __init__(
        self,
        config_yml_filename: Path,
        config_folder: str = ROOT_DIR / "models" / "configuration"
    ):
        config = read_yaml(config_folder / config_yml_filename)
        self.variable = config['data']['variable']
        self.idir = config['data']['idir']
        self.odir = config['data']['odir']
        self.models = config['models']

        logger.info(f'Loading data for variable {self.variable}')
        data_dict = DataLoader(self.variable, self.idir).data_load()
        self.X_train, self.y_train = data_dict['train']
        self.X_test, self.y_test = data_dict['test']

    def run(self):
        # Iterate over each model.
        for i, model in enumerate(self.models):
            self.update_model_output_dir(model['name'])
            logger.info(f'Training and validating model {i+1} '
                         f'out of {len(self.models)}')
            logger.info(f'Training model with method {model["name"]}')
            self.train_and_validate(model)

    def train_and_validate(self, model: Dict):
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
        """ Evaluate the model performance of a model in both trainig and test dataset.
        
        """
        # Model retraining with all training dataset.
        model.fit(self.X_train, self.y_train)
        model.save(self.odir / self.model_name / f"{self.variable}_inception_time.h5")

        # Evaluating performance in test dataset.
        logger.info("Evaluating performance on test set.")
        labels = self.y_test
        preds = model.predict(self.X_test)

        exp_var, maxerr, mae, mse, r2, r2time = get_metric_results(preds, labels)
        # self.save_r2_with_time_structure(r2time, False)
       
        logger.info("Evaluating performance on train set.")
        labels = self.y_train
        preds = model.predict(self.X_train)
        
        # Compute metrics
        tr_exp_var, tr_maxerr, tr_mae, tr_mse, tr_r2, tr_r2time = get_metric_results(
            preds, labels
        )
        # self.save_r2_with_time_structure(tr_r2time, True)

        print(
            f"Exp. Var (test): {tr_exp_var:.4f}({exp_var:.4f})\n"
            f"Max error (test): {tr_maxerr:.4f}({maxerr:.4f})\n"
            f"MAE (test): {tr_mae:.4f}({mae:.4f})\n"
            f"MSE (test): {tr_mse:.4f}({mse:.4f})\n"
            f"R2 (test): {tr_r2:.4f}({r2:.4f})")

        data = {
            'model': self.model_name,
            'variable': self.variable,
            'params': model.get_params(),
            'train': {
                'explained_variance': tr_exp_var,
                'max_errors': tr_maxerr,
                'mean_absolute_error': tr_mae,
                'mean_squared_error': tr_mse,
                'r2': tr_r2
            }, 'test' : {
                'explained_variance': exp_var,
                'max_errors': maxerr,
                'mean_absolute_error': mae,
                'mean_squared_error': mse,
                'r2': r2
            }
        }
        
        # Save results.
        filename = f'allstations_{self.variable}_inception_time'
        logger.debug(f"Saving result of {self.model_name} to {self.results_output_dir}/"
                     f"test_{filename}.yml")
        with open(self.results_output_dir / f"test_{filename}.yml", 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

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

    def update_model_output_dir(self, model_name: str) -> Path:
        self.model_name = model_name
        self.results_output_dir = ROOT_DIR / "models" / "results" / model_name
        os.makedirs(self.results_output_dir, exist_ok=True)


def get_metric_results(preds, labels) -> tuple[float, ...] :
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
    maxerr = float(metrics.max_error(labels, preds))
    mae = float(metrics.mean_absolute_error(labels, preds))
    mse = float(metrics.mean_squared_error(labels, preds))
    r2 = float(metrics.r2_score(labels, preds))    
    ssd = ((labels - preds.reshape(-1, 1)) ** 2).cumsum()
    sst = (labels ** 2).cumsum()
    r2time = (sst - ssd) / sst.iloc[-1]
    return exp_var, maxerr, mae, mse, r2, r2time


if __name__ == '__main__':
    ModelTrain("model_config.yml").run()
