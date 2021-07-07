
import logging 
import numpy as np
import pandas as pd

from src.constants import ROOT_DIR
from dataclasses import dataclass, field
from typing import List, NoReturn, Union
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Concatenate, Add, \
    Activation, Input, GlobalAveragePooling1D, BatchNormalization, Permute
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


logger = logging.getLogger("InceptionTime")


@dataclass
class InceptionTime:
    depth: int = 6
    n_filters: int = 32
    batch_size: int = 64
    n_epochs: int = 200
    inception_kernels: List[int] = field(default_factory=lambda: [2, 4, 8])
    bottleneck_size: int = 32
    verbose: int = 2
    optimizer: str = 'adam'
    loss: str = 'mse'
    metrics: List[str] = field(default_factory=lambda: ['mae'])

    def __post_init__(self) -> NoReturn:
        self.attr_scaler = StandardScaler()
        self.aq_vars_scaler = StandardScaler()
        self.output_directory = ROOT_DIR / "models" / "results" / "InceptionTime"
        self._set_callbacks()

    def _set_callbacks(self):
        logger.info("Two callbacks have been added to the model fitting: "
                    "ModelCheckpoint and ReduceLROnPlateau.")
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                      min_lr=0.0001)
        file_path = self.output_directory / 'best_inceptionTime.hdf5'
        model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='loss',
                                           save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

    def _inception_module(self, input_tensor, stride=1, activation='linear'):
        if int(input_tensor.shape[-2]) > 1:
            input_inception = Conv1D(
                filters=self.bottleneck_size, kernel_size=1, padding='same', 
                activation=activation, use_bias=False
            )(input_tensor)
        else:
            input_inception = input_tensor

        # As presented in original paper InceptionTime: Finding AlexNet for Time Series 
        # Classification. https://arxiv.org/pdf/1909.04939.pdf
        conv_list = []
        for kernel_size in self.inception_kernels:
            conv_list.append(
                Conv1D(
                    filters=self.n_filters, kernel_size=kernel_size, 
                    strides=stride, padding='same', activation=activation,
                    use_bias=False
                )(input_inception)
            )

        max_pool_1 = MaxPool1D(
            pool_size=3, strides=stride, padding='same'
        )(input_tensor)

        conv_6 = Conv1D(
            filters=self.n_filters, kernel_size=1, padding='same', 
            activation=activation, use_bias=False
        )(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=-1)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_inception):
        shortcut_y = Conv1D(
            filters=int(out_inception.shape[-1]), kernel_size=1, padding='same', 
            use_bias=False
        )(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_inception])
        x = Activation('relu')(x)
        return x

    def build_model(self, input_shape: tuple, aux_shape: tuple = None) -> Model:
        logger.debug(f'Input data has shaper {input_shape}')
        input_layer = Input(input_shape)
        
        if aux_shape is not None:
            logger.debug(f'Auxiliary input data has shape {aux_shape}')
            input_aux = Input(aux_shape)

            # Aux layer
            x_aux = Dense(64, activation='relu')(input_aux)
            x_aux = Dense(1, activation='relu')(x_aux)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if d % 3 == 2:
                input_res = x = self._shortcut_layer(input_res, x)

        gap_layer = GlobalAveragePooling1D()(x)
        
        if aux_shape is None:
            concatenated = gap_layer
        else:
            concatenated = Concatenate()([gap_layer, x_aux])
        output_layer = Dense(100, activation='relu')(concatenated)
        output_layer = Dense(1, activation='linear')(output_layer)

        if aux_shape:
            self.model = Model(inputs=[input_layer, input_aux], outputs=output_layer)
        else:
            self.model = Model(inputs=input_layer, outputs=output_layer)

        logger.info(self.model.summary())
        self.model.compile(loss=self.loss, optimizer=self.optimizer, 
                           metrics=['accuracy'])
        return self.model

    def fit(self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        epochs: int = 200
    ):
        features, shapes = self.reshape_data(X)
        self.build_model(*shapes)
        return self.model.fit(
            features, y, epochs=epochs, verbose=self.verbose, callbacks=[self.callbacks])

    def predict(self, X):
        return self.model.predict(self.reshape_data(X, test=True))

    def reshape_data(
        self, 
        X: pd.DataFrame, 
        test: bool = False
    ) -> Union[tuple[pd.DataFrame], tuple[List[pd.DataFrame]]]:
        attr_df = X.filter(regex="_attr$", axis=1)
        aux_df = X.filter(regex="_aux")
        if len(attr_df.columns):
            if test:
                logger.debug("Attribute variables have been scaled with fitted scaler.")
                attr_values = self.attr_scaler.transform(attr_df)
            else:
                logger.debug("Attribute variables have been scaled.")
                attr_values = self.attr_scaler.fit_transform(attr_df)
            aux_values = np.concatenate([attr_values, aux_df.values], axis=1)
        else:
            aux_values = aux_df.values

        # Process temporal feaures. Including scaling ignoring timestep.
        temporal_df = X.filter(regex="_\d+$", axis=1)
        n_time_steps = len(set(map(lambda x: x.split("_")[-1], temporal_df.columns)))
        n_temporal_var = len(temporal_df.columns) // n_time_steps
        temp_values = temporal_df.values.reshape((-1, n_time_steps, n_temporal_var))
        temp_values = temp_values.reshape((-1, n_temporal_var))
        if test:
            logger.debug("Air Quality variables have been scaled with fitted scaler.")
            temp_values = self.aq_vars_scaler.transform(temp_values)
        else:
            logger.debug("Air Quality variables have been scaled.")
            temp_values = self.aq_vars_scaler.fit_transform(temp_values)
        temp_values = temp_values.reshape((-1, n_time_steps, n_temporal_var))

        if aux_values.size:
            logger.info("The input data is separated in temporal features (air quality"
                        " variables) and auxiliary features.")
            return [temp_values, aux_values], \
                (temp_values.shape[1:], aux_values.shape[1:])
        else:
            logger.info("The input data is contains only temporal features (air quality"
                        " variables).")
            return temp_values, temp_values.shape[1:]

    def get_params(self, deep=True):
        return {
            "n_filters": self.n_filters, "bottleneck_size": self.bottleneck_size,
            "optimizer": self.optimizer, "loss": self.loss, 
            'batch_size': self.batch_size, "n_epochs": self.n_epochs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.__post_init__()
        return self

    def save(self, filename: str):
        self.model.save(self.output_directory / f"{filename}.h5")

    def load(self, filename: str):
        self.model = load_model(self.output_directory / filename)
        print(self.model.summary())
