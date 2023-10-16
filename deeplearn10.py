import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score #정밀도, 재현률

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 사용자 입력 값 받기 = dataset
#dataset_addr = input("csv 파일을 선택하세요 : ")
dataset = pd.read_csv('kddcup.csv') # need user input
#dataset = pd.read_csv(dataset_addr)
print("데이터셋 처음 다섯 개 조회 : ",dataset.head())

# 'normal.' 컬럼의 고유 값 확인
print("normal. 컬럼의 고유값 확인 : ",dataset['normal.'].unique())


# 'normal.' 이외의 모든 카테고리를 대체할 목록을 생성
categories_to_replace = [category for category in dataset['normal.'].unique() if category != 'normal.']

# Except 'normal.', all categories change into 'attack'.
dataset['normal.'] = dataset['normal.'].replace(categories_to_replace, 'attack')

print("변경 후 normal. 컬럼의 고유 값 확인 : ",dataset['normal.'].unique())


# # x = 마지막 label 열을 제외한 모든 열을 특징으로 사용
# # y = 마지막 label 열을 카테고리로 사용
x = dataset.iloc[:, :dataset.shape[1]-1].values
y = dataset.iloc[:, dataset.shape[1]-1].values

# print(x.shape, y.shape)

# uniq1 = dataset.tcp.unique()
# uniq2 = dataset.http.unique()
# uniq3 = dataset.SF.unique()

# print(uniq1, '\n', uniq2, '\n', uniq3)
# print(uniq1.size, '\n', uniq2.size, '\n', uniq3.size)

#문자열을 포함하는 컬럼을 동적으로 찾기
string_columns = dataset.select_dtypes(include=['object']).columns

unique_values = {}
for column in string_columns:
    unique_vals = dataset[column].unique()
    unique_values[column] = unique_vals
    print(f"{column}의 고유 값: {unique_vals}")
    print(f"{column}의 고유 값 개수: {len(unique_vals)}\n")

print(f"x shape: {x.shape}, y shape: {y.shape}")

from sklearn.compose import ColumnTransformer
string_columns_idx = [idx for idx, dtype in enumerate(x[0]) if isinstance(dtype, str)]
label_encoders = [LabelEncoder() for _ in string_columns_idx]
column_transformers = []
original_x_shape = x.shape[1]
    
for i, idx in enumerate(string_columns_idx):
    x[:, idx] = label_encoders[i].fit_transform(x[:, idx])
    column_transformers.append(("encoder_"+str(idx), OneHotEncoder(), [idx]))
    
ct = ColumnTransformer(column_transformers, remainder='passthrough')
x = ct.fit_transform(x)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
    
print(f"Original x shape: {original_x_shape}, \nTransformed x shape: {x.shape}")
print(f"Transformed y shape: {y.shape}")


import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
seed = 0
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

model = Sequential()
model.add(Dense(30, input_dim=x.shape[1], activation='relu'))
# fully connected layer
model.add(Dense(1, activation='sigmoid')) #1단 딥러닝

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=50) #학습

print("\n Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))