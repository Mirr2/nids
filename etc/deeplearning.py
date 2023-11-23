import datetime
before_time = datetime.datetime.now()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

dataset = pd.read_csv('./kddcup.csv')
#dataset = pd.read_csv('./kddcup.data_10_percent_corrected.csv')
print(dataset.head())
print(dataset['normal.'].unique())

dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')
print("normal을 제외한 모든 공격카테고리를 attack으로 변경 : ",dataset['normal.'].unique(),"\n")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 41].values
print(x.shape, y.shape)

uniq1 = dataset.tcp.unique()
uniq2 = dataset.http.unique()
uniq3 = dataset.SF.unique()

print("uniq1 =",uniq1,"\nuniq1 Size :",uniq1.size, "\nuniq2 =",uniq2,"\nuniq2 Size :",uniq2.size,"\nuniq3 =",uniq3,"\nuniq3 Size :",uniq3.size)


from sklearn.compose import ColumnTransformer
# label encoder를 통해 문자를 숫자로 바꿔줌.
labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()

x[:, 1] = labelencoder_x_1.fit_transform(x[:,1])
x[:, 2] = labelencoder_x_2.fit_transform(x[:,2])
x[:, 3] = labelencoder_x_3.fit_transform(x[:,3])

onehotencoder_1 = ColumnTransformer([("tcp", OneHotEncoder(), [1])], remainder = 'passthrough')
onehotencoder_2 = ColumnTransformer([("http", OneHotEncoder(), [4])], remainder= 'passthrough')
onehotencoder_3 = ColumnTransformer([("SF", OneHotEncoder(), [74])], remainder= 'passthrough')

x = np.array(onehotencoder_1.fit_transform(x))
x = np.array(onehotencoder_2.fit_transform(x))
x = np.array(onehotencoder_3.fit_transform(x))

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print(x.shape, y.shape)

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

print(len(x_train), len(x_test))
print(len(y_train), len(y_test))

seed = 0
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

x_train =  np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

model = Sequential()
model.add(Dense(30, input_dim=122, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=50)
print("\n Accuracy: %.4f" %(model.evaluate(x_test,y_test)[1]))

y_pred = (model.predict(x_test, verbose=1) > 0.5).astype("int32")
cnf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy : ", acc, "/ F1-Score : ", f1)
print(cnf_matrix)

#작업 소요 시간
last_time = datetime.datetime.now()
last_time = last_time-before_time
print("작업 소요 시간:", last_time)