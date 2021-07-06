from src.constants import ROOT_DIR
from dataclasses import dataclass, field
from typing import List, NoReturn
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Concatenate, Add, \
    Activation, Input, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



@dataclass
class InceptionTime:
    depth: int = 6
    n_filters: int = 32
    batch_size: int = 64
    n_epochs: int = 200
    bottleneck_size: int = 32
    optimizer: str = 'adam'
    loss: str = 'mse'
    metrics: List[str] = field(default_factory=lambda: ['mae'])

    def __post_init__(self) -> NoReturn:
        self.output_directory = ROOT_DIR / "models" / "results" / "InceptionTime"
        self._set_callbacks()

    def _set_callbacks(self):
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                      min_lr=0.0001)
        file_path = self.output_directory / 'best_inceptionTime.hdf5'
        model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='loss',
                                           save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

    def _inception_module(self, input_tensor, stride=1, activation='linear'):
        if int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(
                filters=self.bottleneck_size, kernel_size=1, padding='same', 
                activation=activation, use_bias=False
            )(input_tensor)
        else:
            input_inception = input_tensor

        # As presented in original paper InceptionTime: Finding AlexNet for Time Series 
        # Classification. https://arxiv.org/pdf/1909.04939.pdf
        kernel_size_s = [10, 20, 40]

        conv_list = []
        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(
                filters=self.n_filters, kernel_size=kernel_size_s[i], strides=stride, 
                padding='same', activation=activation, use_bias=False
            )(input_inception))

        max_pool_1 = MaxPool1D(
            pool_size=3, strides=stride, padding='same'
        )(input_tensor)

        conv_6 = Conv1D(
            filters=self.n_filters, kernel_size=1, padding='same', 
            activation=activation, use_bias=False
        )(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
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

    def build_model(self, input_shape):
        input_layer = Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                input_res = x = self._shortcut_layer(input_res, x)

        gap_layer = GlobalAveragePooling1D()(x)
        output_layer = Dense(100, activation='relu')(gap_layer)
        output_layer = Dense(1, activation='linear')(output_layer)

        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, 
                           metrics=['accuracy'])
        return self.model

    def fit(self,
        X,
        y,
        epochs: int = 200
    ):
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        n_labels = y.shape[1] if len(y.shape) > 1 else 1
        self.compile(n_features, n_labels)
        return self.model.fit(X, y, epochs=epochs, verbose=2, 
                              callbacks=[self.callbacks])

    def predict(self, X):
        return self.model.predict(X)

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
