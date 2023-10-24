import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense

class DataPreprocessor:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.x = None
        self.y = None

    def preprocess_data(self):
        self._replace_categories()
        self._split_features_labels()
        self._encode_string_columns()

    def _replace_categories(self):
        categories_to_replace = [category for category in self.dataset['normal.'].unique() if category != 'normal.']
        self.dataset['normal.'] = self.dataset['normal.'].replace(categories_to_replace, 'attack')

    def _split_features_labels(self):
        self.x = self.dataset.iloc[:, :self.dataset.shape[1]-1].values
        self.y = self.dataset.iloc[:, self.dataset.shape[1]-1].values

    def _encode_string_columns(self):
        string_columns_idx = [idx for idx, dtype in enumerate(self.x[0]) if isinstance(dtype, str)]
        label_encoders = [LabelEncoder() for _ in string_columns_idx]
        column_transformers = []

        for i, idx in enumerate(string_columns_idx):
            self.x[:, idx] = label_encoders[i].fit_transform(self.x[:, idx])
            column_transformers.append(("encoder_"+str(idx), OneHotEncoder(), [idx]))

        ct = ColumnTransformer(column_transformers, remainder='passthrough')
        self.x = ct.fit_transform(self.x)
        labelencoder_y = LabelEncoder()
        self.y = labelencoder_y.fit_transform(self.y)


class ModelTrainer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = None

    def train_model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
        seed = 0
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        x_train = np.asarray(x_train).astype('float32')
        x_test = np.asarray(x_test).astype('float32')
        y_test = np.asarray(y_test).astype('float32')

        self.model = Sequential()
        self.model.add(Dense(30, input_dim=self.x.shape[1], activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=10, batch_size=50)

        return x_test, y_test


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)[1]


if __name__ == "__main__":
    data_preprocessor = DataPreprocessor('kddcup.csv')
    data_preprocessor.preprocess_data()

    model_trainer = ModelTrainer(data_preprocessor.x, data_preprocessor.y)
    x_test, y_test = model_trainer.train_model()

    model_evaluator = ModelEvaluator(model_trainer.model)
    accuracy = model_evaluator.evaluate_model(x_test, y_test)

    print("\n Accuracy: %.4f" % accuracy)