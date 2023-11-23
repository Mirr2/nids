from keras.callbacks import Callback
from PyQt5.QtCore import pyqtSignal, QObject


class CustomCallback(Callback, QObject):
    update_progress_signal = pyqtSignal(int)
    update_graph_signal = pyqtSignal(list, list)  # 그래프 업데이트 신호

    def __init__(self, progress_bar, mplwidget):
        Callback.__init__(self)
        QObject.__init__(self)
        self.progress_bar = progress_bar
        self.mplwidget = mplwidget
        self.update_progress_signal.connect(self.progress_bar.setValue)
        self.update_graph_signal.connect(self.mplwidget.update_graph)  # 그래프 업데이트 연결

        self.loss_data = []  # 손실 데이터 초기화
        self.acc_data = []  # 정확도 데이터 초기화

    def on_epoch_end(self, epoch, logs=None):
        # 프로그레스 바 업데이트
        progress_value = 70 + 30 * (epoch + 1) / self.params['epochs']
        self.update_progress_signal.emit(int(progress_value))

        # 손실과 정확도 데이터 저장
        self.loss_data.append(logs['loss'])
        self.acc_data.append(logs['accuracy'])

        self.update_graph_signal.emit(self.loss_data, self.acc_data)