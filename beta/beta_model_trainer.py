from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from keras.callbacks import History

from beta_callback import CustomCallback


class ModelTrainer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = None

    def train_model_with_activation(self, activation_function='relu', output_layer_type='sigmoid', callbacks=None):
        if callbacks is None:
            callbacks = []
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed)

        x_train = np.asarray(x_train).astype('float32')
        x_test = np.asarray(x_test).astype('float32')
        y_test = np.asarray(y_test).astype('float32')

        self.model = Sequential()
        self.model.add(Dense(30, input_dim=self.x.shape[1], activation=activation_function))

        if output_layer_type == 'sigmoid':
            self.model.add(Dense(1, activation='sigmoid'))
        elif output_layer_type == 'softmax':
            if len(self.y.shape) == 1:
                self.model.add(Dense(1, activation='softmax'))
            else:
                self.model.add(Dense(self.y.shape[1], activation='softmax'))
        elif output_layer_type == 'linear':
            self.model.add(Dense(1, activation='linear'))
        else:
            raise ValueError('Invalid output_layer_type')

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = History()
        if callbacks:
            callbacks.append(history)
        else:
            callbacks = [history]

        self.model.fit(x_train, y_train, epochs=10, batch_size=50)

        return self.model, x_test, y_test, history.history['accuracy'], history.history['loss']

# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense
# import numpy as np
# import tensorflow as tf

# class ModelTrainer:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.model = None

#     def train_model_with_activation(self, activation_function='relu'):
#         x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
#         seed = 0
#         np.random.seed(seed)
#         tf.compat.v1.set_random_seed(seed)

#         x_train = np.asarray(x_train).astype('float32')
#         x_test = np.asarray(x_test).astype('float32')
#         y_test = np.asarray(y_test).astype('float32')

#         self.model = Sequential()
#         self.model.add(Dense(30, input_dim=self.x.shape[1], activation=activation_function))
#         self.model.add(Dense(1, activation='sigmoid'))

#         self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         self.model.fit(x_train, y_train, epochs=10, batch_size=50)

#         return x_test, y_test

#=======================

    # def train_model(self):
    #     x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
    #     seed = 0
    #     np.random.seed(seed)
    #     tf.compat.v1.set_random_seed(seed)

    #     x_train = np.asarray(x_train).astype('float32')
    #     x_test = np.asarray(x_test).astype('float32')
    #     y_test = np.asarray(y_test).astype('float32')

    #     self.model = Sequential()
    #     self.model.add(Dense(30, input_dim=self.x.shape[1], activation='relu'))
    #     self.model.add(Dense(1, activation='sigmoid'))

    #     self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     self.model.fit(x_train, y_train, epochs=10, batch_size=50)

    #     return x_test, y_test