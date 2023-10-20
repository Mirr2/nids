import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import test
import numpy as np

# X와 Y를 훈련 데이터(train)와 테스트 데이터(test)로 분류 = 70 : 30
x_train, x_test, y_train, y_test = train_test_split(x, y,
                   test_size = 0.3, random_state = 0) #148206 테스트용 데이터
#345814=train
#148206=test

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

seed = 0
np.random.seed(seed)
#tf.set_random_seed(seed)
tf.compat.v1.set_random_seed(seed)

x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

model = Sequential()
model.add(Dense(30, input_dim=118, activation='relu'))
# fully connected layer
model.add(Dense(1, activation='sigmoid')) #1단 딥러닝

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=50) #학습

print("\n Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))
"""
y_test = np.ndarray.reshape(y_test,len(y_test))
print(y_test.dtype)
print(y_test[100])
"""
#y_pred = model.predict_classes(x_test) #labelling 0,1
#y_pred = model.predict(x_test)
y_pred = (model.predict(x_test, verbose=1) > 0.5).astype("int32") #sigmoid인 경우

#y_pred = np.argmax(model.predict(x), axis=-1) #softmax인 경우

"""
for t in y_pred:
    if t != 1.0 and t != 0.0:
        print(t)

y_pred = np.ndarray.reshape(y_pred,len(y_test))
print(y_pred.dtype)
print(y_pred[100])
"""
# 혼돈 행렬과 정확도, F1 점수로 모델 평가
cnf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy : ", acc, "/ F1-Score : ", f1)
print(cnf_matrix)