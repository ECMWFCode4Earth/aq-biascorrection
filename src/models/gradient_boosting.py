import logging 
import numpy as np
import pandas as pd
import tensorflow as tf

from src.constants import ROOT_DIR
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NoReturn, Union
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Concatenate, Add, \
    Activation, Input, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


logger = logging.getLogger("InceptionTime")

NUM_SAMPLES = 100000


class GradientBoosting:
    def __init__(self, n_bathces: int = 1):
        self.n_batches = n_batches
        

    def build_model(self, df: pd.DataFrame):
        numerical_columns = []
        categorical_columns = []
        for col_name in df.columns:
            if ('hour' in col_name) or ('month' in col_name):
                categorical_columns.append(col_name)
            else:
                numerical_columns.append(col_name)
        
        feature_columns = []
        for feature_name in CATEGORICAL_COLUMNS:
            # Need to one-hot encode categorical features.
            vocabulary = dftrain[feature_name].unique()
            feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

        for feature_name in NUMERIC_COLUMNS:
            feature_columns.append(
                tf.feature_column.numeric_column(feature_name, dtype=tf.float32)
            )
        self.model = tf.estimator.BoostedTreesClassifier(
            feature_columns, n_batches_per_layer=n_batches
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, valiadtion_split: float = 0.8):
        self.build_model(X)
        # TODO: implement validation split
        X_val, y_val = pd.DataFrame, pd.DataFrame
        train_input_fn = make_input_funtion(X, y)
        eval_input_fn = make_input_funtion(
            X_val, y_val, shuffle=False, n_epochs=self.n_epochs)
        self.model.train(train_input_fn, max_steps=1)
        result = self.model.evaluate(eval_input_fn)
        print(pd.Series(result))


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    
def make_input_function(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    n_epochs: int = None,
    shuffle: bool = True
) -> Callable:
    def func():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = dataset.repeat(n_epochs)
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return func
