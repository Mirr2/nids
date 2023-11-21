from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf

# CustomCallback을 import합니다. (경로는 실제 CustomCallback의 위치에 따라 달라질 수 있습니다.)
from beta_callback import CustomCallback

class ModelTrainer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = None

    def train_model_with_activation(self, activation_function='relu', output_layer_type='sigmoid', progress_bar=None, mplwidget=None):
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

        # CustomCallback을 생성하고, 진행 상태를 업데이트할 위젯을 전달합니다.
        custom_callback = CustomCallback(progress_bar=progress_bar, mplwidget=mplwidget)

        # 모델 훈련 시, CustomCallback을 콜백으로 추가합니다.
        self.model.fit(x_train, y_train, epochs=10, batch_size=50, callbacks=[custom_callback])

        # 훈련 후의 손실과 정확도를 custom_callback 객체에서 가져옵니다.
        accuracy = custom_callback.acc_data
        loss = custom_callback.loss_data

        return self.model, x_test, y_test, accuracy, loss
