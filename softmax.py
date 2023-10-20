import sys
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from onehot import onehotencode
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def deep_learning(file_path):
    df = pd.read_csv(file_path)
    x, y = onehotencode(df)  # onehotencode로 전처리한 경우

    # X와 Y를 훈련 데이터(train)와 테스트 데이터(test)로 분류 (70 : 30)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    seed = 0
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

    x_train = np.asarray(x_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    model = Sequential()
    model.add(Dense(30, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=50)

    accuracy = model.evaluate(x_test, y_test)[1]
    print("Accuracy: %.4f" % accuracy)

    y_pred = (model.predict(x_test, verbose=1) > 0.5).astype("int32")

    cnf_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    print("Accuracy: ", acc, "/ F1-Score: ", f1)
    print(cnf_matrix)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python softmax.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    deep_learning(file_path)
