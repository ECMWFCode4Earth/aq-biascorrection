import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NoReturn

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.constants import ROOT_DIR
from src.logging import get_logger

logger = get_logger("Gradient Boosting")

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) == 0:
    logger.info("Not enough GPU hardware devices available")
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logger.info(f"A total of {len(physical_devices)} GPU devices are available.")

NUM_SAMPLES = 100000


class GradientBoosting:
    def __init__(self,
                 n_batches: int = 1,
                 n_epochs: int = 50,
                 output_dims: int = 24,
                 num_samples: int = 100000):
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.num_samples = num_samples
        self.output_dims = output_dims

    def build_model(self, df: pd.DataFrame) -> NoReturn:
        numerical_columns = []
        categorical_columns = []
        for col_name in df.columns:
            if ('hour' in col_name) or ('month' in col_name):
                numerical_columns.append(col_name)
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
            validation_split: float = 0.8) -> NoReturn:
        self.build_model(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          test_size=validation_split)
        train_input_fn = self.make_input_function(X, y)
        eval_input_fn = self.make_input_function(
            X_val,
            y_val,
            shuffle=False,
            n_epochs=self.n_epochs,
            num_samples=self.num_samples
        )
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
        shuffle: bool = True,
        num_samples: int = 100000
    ) -> Callable:
        def func():
            dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
            if shuffle:
                dataset = dataset.shuffle(num_samples)
            # For training, cycle thru dataset as many times as need (n_epochs=None).
            dataset = dataset.repeat(n_epochs)
            # In memory training doesn't use batching.
            dataset = dataset.batch(num_samples)
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
