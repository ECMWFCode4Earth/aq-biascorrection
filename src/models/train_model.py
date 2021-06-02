from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from src.data.load.load_data import DataLoader

import xgboost as xg

import warnings

warnings.filterwarnings('ignore')


class ModelTrain:
    def __init__(
            self,
            variable: str,
            input_dir: Path
    ):
        data_loader = DataLoader(variable, input_dir)
        self.data_dict = data_loader.data_load()
        self.model = xg.XGBRegressor(verbosity=1)

    def train(self):
        parameters = {'nthread': [-1],
                      # when use hyperthread, xgboost may become slower
                      'objective': ['reg:squarederror'],
                      'learning_rate': [0.1, 0.2, 0.5, 0.7, 1],
                      # so called `eta` value
                      'max_depth': [5, 6, 7, 8],
                      'min_child_weight': [4, 5, 6, 7, 8],
                      'subsample': [0.7],
                      'colsample_bytree': [0.7],
                      'n_estimators': [500, 700]}

        xgb_grid = GridSearchCV(self.model,
                                parameters,
                                cv=5,
                                scoring='neg_mean_absolute_error',
                                n_jobs=1,
                                verbose=10)
        xgb_grid.fit(self.data_dict['train'][0],
                     self.data_dict['train'][1])

        # Print the r2 score
        print(r2_score(self.data_dict['test'][1],
                       xgb_grid.best_estimator_.predict(
                           self.data_dict['test'][0]
                       )))

        print(mean_squared_error(self.data_dict['test'][1],
                                 xgb_grid.best_estimator_.predict(
                                 self.data_dict['test'][0]
                                 )))


if __name__ == '__main__':
    ModelTrain('pm25', Path('../../data/processed/')).train()
