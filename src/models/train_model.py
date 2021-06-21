import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import xgboost as xg
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from src.data.load.load_data import DataLoader
from src.models.utils import read_yaml
from src.constants import ROOT_DIR

warnings.filterwarnings('ignore')

logger = logging.getLogger("Model trainer")

models_dict = {
    'xgboost_regressor': xg.XGBRegressor()
}


class ModelTrain:
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
            logger.info(f'Training and validating model {i+1} '
                         f'out of {len(self.models)}')
            logger.info(f'Training model with method {model["name"]}')
            gridsearch = self.train_and_validate(model)
            model_output_path = self.get_model_output_path(gridsearch, model)

    def train_and_validate(self, model: Dict):
        training_params = model['training_method']

        gridsearch = GridSearchCV(
            models_dict[model['type']],
            model['model_parameters'],
            cv=training_params['cv'],
            scoring=training_params['scoring'],
            n_jobs=training_params['n_jobs'],
            verbose=training_params['verbose'])

        gridsearch.fit(self.X_train, self.y_train)
        self.evaluate_model(gridsearch.best_estimator_)

        return gridsearch
    
    def evaluate_model(
        self, 
        model, 
        retrain: bool = True, 
        test: bool = True
    ) -> Tuple[float, float, float, float, float]:
        if retrain:
            model.fit(self.X_train, self.y_test)

        if test:
            logger.info("Evaluating performance on test set.")
            labels = self.y_test
            preds = model.predict(self.X_test)
        else:
            logger.info("Evaluating performance on train set.")
            labels = self.y_train
            preds = model.predict(self.X_train)
        
        # Compute metrics
        exp_var = float(metrics.explained_variance_score(labels, preds))
        maxerr = float(metrics.max_error(labels, preds))
        mae = float(metrics.mean_absolute_error(labels, preds))
        mse = float(metrics.mean_squared_error(labels, preds))
        r2 = float(metrics.r2_score(labels, preds))

        print(f"Exp. Var: {exp_var:.4f}\nMax error: {maxerr:.4f}\n"
              f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR2: {r2:.4f}")

        return exp_var, maxerr, mae, mse, r2

    def get_model_output_path(self, gridsearch, model) -> Path:
        pass


if __name__ == '__main__':
    ModelTrain("model_config.yml").run()
