from pathlib import Path
from typing import Dict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from src.data.load.load_data import DataLoader
from src.models.utils import read_yaml

import xgboost as xg

import warnings

warnings.filterwarnings('ignore')

models_dict = {
    'xgboostregressor': xg.XGBRegressor()
}


class ModelTrain:
    def __init__(
            self,
            config_yml: Path
    ):
        self.config = read_yaml(config_yml)

    def run(self):
        for model in self.config['models']:
            data_dict = DataLoader(model['variable'],
                                   model['data_dir']).data_load()
            gridsearch = self.train(model, data_dict)
            model_output_path = self.get_model_output_path(gridsearch, model)

    def train(self, model: Dict, data: Dict):
        model_instance = models_dict[model['type']]
        parameters = model['model_parameters'][0]
        training_methods = model['training_method'][0]

        xgb_grid = GridSearchCV(model_instance,
                                parameters,
                                cv=training_methods['cv'],
                                scoring=training_methods['scoring'],
                                n_jobs=training_methods['n_jobs'],
                                verbose=training_methods['verbose'])
        xgb_grid.fit(data['train'][0],
                     data['train'][1])

        # Print the r2 score
        print(r2_score(data['test'][1],
                       xgb_grid.best_estimator_.predict(
                           data['test'][0]
                       )))

        print(mean_squared_error(data['test'][1],
                                 xgb_grid.best_estimator_.predict(
                                 data['test'][0]
                                 )))
        return xgb_grid

    def get_model_output_path(self, gridsearch, model) -> Path:
        pass


if __name__ == '__main__':
    ModelTrain('pm25', Path('../../data/processed/')).train()
