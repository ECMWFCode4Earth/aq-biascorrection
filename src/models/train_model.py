import logging
import warnings
from pathlib import Path
from typing import Dict

import xgboost as xg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    def run(self):
        data_dict = DataLoader(self.variable, self.idir).data_load()
        for i, model in enumerate(self.models):
            logger.info(f'Training and validating model {i+1} '
                         f'out of {len(self.config["models"])}')
            logger.info(f'Loading data for variable {model["variable"]}')
            
            logger.info(f'Training model with method {model["type"]}')
            gridsearch = self.train_and_validate(model, data_dict)
            model_output_path = self.get_model_output_path(gridsearch, model)

    def train_and_validate(self, model: Dict, data: Dict):
        model_instance = models_dict[model['type']]
        parameters = model['model_parameters'][0]
        training_methods = model['training_method'][0]

        gridsearch = GridSearchCV(
            model_instance,
            parameters,
            cv=training_methods['cv'],
            scoring=training_methods['scoring'],
            n_jobs=training_methods['n_jobs'],
            verbose=training_methods['verbose'])

        print(data['train'][1].mean().values)
        gridsearch.fit(data['train'][0], data['train'][1])

        # Print the r2 score
        print(r2_score(data['test'][1],
                       gridsearch.best_estimator_.predict(
                           data['test'][0]
                       )))

        print(mean_absolute_error(data['test'][1],
                                  gridsearch.best_estimator_.predict(
                                  data['test'][0]
                                  )))
        return gridsearch

    def get_model_output_path(self, gridsearch, model) -> Path:
        pass


if __name__ == '__main__':
    ModelTrain("model_config.yml").run()
