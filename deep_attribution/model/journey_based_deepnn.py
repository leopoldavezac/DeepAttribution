
from tensorflow.keras.layers import (
    Dense, LSTM, Input, Lambda, RepeatVector, Permute, Flatten, Activation, Multiply
)
from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy

from numpy import ndarray

class JourneyBasedDeepNN:

    def __init__(
        self,
        max_nb_eng_per_journey: int,
        n_cmpgns: int,
        epochs: int = 5,
        n_hidden_units_embedding: int = 20,
        n_hidden_units_lstm: int = 64,
        dropout_lstm: float = 0.1,
        recurrent_dropout_lstm: float = 0.1,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7
        ) -> None:

        self.__max_nb_eng_per_journey = max_nb_eng_per_journey
        self.__n_cmpgns = n_cmpgns
        self.__epochs = epochs

        self.__create_input_layer()
        self.__create_cmpgn_embedding_layer(n_hidden_units_embedding)
        self.__create_lstm_layer(n_hidden_units_lstm, dropout_lstm, recurrent_dropout_lstm)
        self.__create_attention_layer()
        self.__create_weighted_activation_layer(n_hidden_units_lstm)
        self.__create_output_layer()

        self.__nn = Model(inputs=self.__input_layer, outputs=self.__output_layer)

        opt = Adam(
            learning_rate,
            beta_1,
            beta_2,
            epsilon
        )

        metrics = [
            Accuracy(name="accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc")
        ]

        self.__nn.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=metrics
            )


    def __create_input_layer(self) -> None:

        self.__input_layer = Input(
            shape=(self.__max_nb_eng_per_journey, self.__n_cmpgns)
            )

    def __create_cmpgn_embedding_layer(self, n_hidden_units: int) -> None:

        self.__cmpgn_embedding_layer = Dense(
            n_hidden_units,
            activation="linear",
            input_shape=(self.__max_nb_eng_per_journey, self.__n_cmpgns)
            )(self.__input_layer)

    def __create_lstm_layer(
        self,
        n_hidden_units: int,
        dropout: float,
        recurent_dropout: float
        ) -> None:

        self.__lstm_layer = LSTM(
            n_hidden_units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurent_dropout
            )(self.__cmpgn_embedding_layer)


    def __create_attention_layer(self) -> None:

        self.__attention_layer = Dense(1, activation="tanh")(self.__lstm_layer)
        self.__attention_layer = Flatten()(self.__attention_layer)
        self.__attention_layer = Activation('softmax', name='attribution')(self.__attention_layer)

    def __create_weighted_activation_layer(self, n_hidden_units_lstm: int) -> None:

        self.__attention_layer = RepeatVector(1)(self.__attention_layer)
        self.__attention_layer = Permute([2, 1])(self.__attention_layer)

        self.__weighted_activation_layer = Multiply()([self.__lstm_layer, self.__attention_layer])
        self.__weighted_activation_layer = Lambda(
            lambda xin: backend.sum(xin, axis=-2), 
            output_shape=(n_hidden_units_lstm,)
            )(self.__weighted_activation_layer)

    def __create_output_layer(self) -> None:

        self.__output_layer = Dense(1, activation="sigmoid")(self.__weighted_activation_layer)


    def fit(
        self,
        X: ndarray,
        y: ndarray,
        batch_size: int = 64,
        ) -> None:

        self.__nn.fit(
            X,
            y,
            batch_size,
            self.__epochs
        )


    def batch_fit(
        self,
        train_batch_loader: Sequence,
        test_batch_loader: Sequence,
        ) -> None:
        
        
        self.__nn.fit(
            x = train_batch_loader,
            validation_data = test_batch_loader,
            epochs = self.__epochs
            )

    def predict(
        self,
        X: ndarray
        ) -> None:

        self.__nn.predict(X)


    def evaluate(
        self,
        X,
        y
        ) -> float:

        return self.__nn.evaluate(X, y)

    
    def batch_evaluate(self, batch_loader: Sequence) -> ndarray:

        return self.__nn.evaluate(batch_loader)


    def save_attention_model(self, model_dir: str) -> None:

        attention_model = Model(
            inputs=self.__nn.input,
            outputs=self.__nn.get_layer("attribution").output
            )

        attention_model.save(model_dir)
        


    def save_model(self, model_dir: str) -> None:

        self.__nn.save(model_dir)