import logging 
import numpy as np
import pandas as pd
import tensorflow as tf

from src.constants import ROOT_DIR
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NoReturn, Union, Callable
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Concatenate, Add, \
    Activation, Input, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


logger = logging.getLogger("InceptionTime")

NUM_SAMPLES = 100000


class GradientBoosting:
    def __init__(self,
                 n_batches: int = 1,
                 n_epochs: int = 50):
        self.n_batches = n_batches
        self.n_epochs = n_epochs

    def build_model(self, df: pd.DataFrame):
        numerical_columns = []
        categorical_columns = []
        for col_name in df.columns:
            if ('hour' in col_name) or ('month' in col_name):
                categorical_columns.append(col_name)
            else:
                numerical_columns.append(col_name)
        
        feature_columns = []
        for feature_name in categorical_columns:
            # Need to one-hot encode categorical features.
            vocabulary = df[feature_name].unique()
            feature_columns.append(
                self.one_hot_cat_column(feature_name, vocabulary)
            )

        for feature_name in numerical_columns:
            feature_columns.append(
                tf.feature_column.numeric_column(feature_name, dtype=tf.float32)
            )
        self.model = tf.estimator.BoostedTreesClassifier(
            feature_columns,
            n_batches_per_layer=self.n_batches
        )

    def fit(self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            validation_split: float = 0.8):
        self.build_model(X)
        # TODO: implement validation split
        X_val, y_val = pd.DataFrame, pd.DataFrame
        train_input_fn = self.make_input_function(X, y)
        eval_input_fn = self.make_input_function(
            X_val, y_val,
            shuffle=False,
            n_epochs=self.n_epochs)
        self.model.train(train_input_fn, max_steps=100)
        result = self.model.evaluate(eval_input_fn)
        print(pd.Series(result))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def make_input_function(
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_epochs: int = None,
        shuffle: bool = True
    ) -> Callable:
        def func():
            dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
            if shuffle:
                dataset = dataset.shuffle(NUM_SAMPLES)
            # For training, cycle thru dataset as many times as need (n_epochs=None).
            dataset = dataset.repeat(n_epochs)
            # In memory training doesn't use batching.
            dataset = dataset.batch(NUM_SAMPLES)
            return dataset
        return func

    @staticmethod
    def one_hot_cat_column(feature_name, vocab):
        return tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                feature_name,
                vocab
            )
        )
