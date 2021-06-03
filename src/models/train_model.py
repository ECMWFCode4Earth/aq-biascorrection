from pathlib import Path
from typing import Dict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.data.load.load_data import DataLoader
from src.models.utils import read_yaml

import xgboost as xg

import warnings
import logging

warnings.filterwarnings('ignore')

models_dict = {
    'xgboost_regressor': xg.XGBRegressor()
}


class ModelTrain:
    def __init__(
            self,
            config_yml: Path
    ):
        self.config = read_yaml(config_yml)

    def run(self):
        for i, model in enumerate(self.config['models']):
            logging.info(f'Training and validating model {i+1} '
                         f'out of {len(self.config["models"])}')
            logging.info(f'Loading data for variable {model["variable"]}')
            data_dict = DataLoader(model['variable'],
                                   Path(model['data_dir'])).data_load()
            logging.info(f'Training model with method {model["type"]}')
            gridsearch = self.train_and_validate(model, data_dict)
            model_output_path = self.get_model_output_path(gridsearch, model)

    def train_and_validate(self, model: Dict, data: Dict):
        model_instance = models_dict[model['type']]
        parameters = model['model_parameters'][0]
        training_methods = model['training_method'][0]

        gridsearch = GridSearchCV(model_instance,
                                parameters,
                                cv=training_methods['cv'],
                                scoring=training_methods['scoring'],
                                n_jobs=training_methods['n_jobs'],
                                verbose=training_methods['verbose'])

        print(data['train'][1].mean().values)
        gridsearch.fit(data['train'][0],
                     data['train'][1])

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
    ModelTrain(Path('../../models/configuration/model_config.yml')).run()
