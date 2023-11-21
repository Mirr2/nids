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


# from keras.callbacks import Callback
# from PyQt5.QtCore import pyqtSignal, QObject
#
#
# class CustomCallback(Callback, QObject):
#     update_signal = pyqtSignal(str)
#     update_progress_signal = pyqtSignal(int)
#     update_graph_signal = pyqtSignal(list, list)  # 그래프 업데이트 신호 추가
#
#     def __init__(self, text_edit, progress_bar, mplwidget):
#         Callback.__init__(self)
#         QObject.__init__(self)
#         self.text_edit = text_edit
#         self.progress_bar = progress_bar
#         self.mplwidget = mplwidget
#         self.update_signal.connect(self.update_text_edit)
#         self.update_progress_signal.connect(self.update_progress_bar)
#         self.update_graph_signal.connect(self.mplwidget.update_graph)  # 그래프 업데이트 연결
#
#         self.loss_data = []  # 손실 데이터 초기화
#         self.acc_data = []  # 정확도 데이터 초기화
#
#     def on_epoch_end(self, epoch, logs=None):
#         text = f"Epoch {epoch + 1}/{self.params['epochs']} - " \
#                f"loss: {logs['loss']:.4f} - " \
#                f"accuracy: {logs['accuracy']:.4f}\n"
#         self.update_signal.emit(text)
#         # 프로그레스 바 업데이트 (70 + 30 * (현재 에폭 + 1) / 전체 에폭 수)
#         progress_value = 70 + 30 * (epoch + 1) / self.params['epochs']
#         self.update_progress_signal.emit(int(progress_value))
#         # 그래프 데이터를 업데이트하기 위해 값을 저장합니다.
#         self.loss_data.append(logs['loss'])
#         self.acc_data.append(logs['accuracy'])
#
#         # 실시간으로 그래프를 업데이트하기 위한 신호를 보냅니다.
#         self.update_graph_signal.emit(self.loss_data, self.acc_data)
#
#     def on_batch_end(self, batch, logs=None):
#         text = f"Batch {batch + 1}/{self.params['steps']} - " \
#                f"loss: {logs['loss']:.4f} - " \
#                f"accuracy: {logs['accuracy']:.4f}\n"
#         self.update_signal.emit(text)
#
#     def update_text_edit(self, text):
#         self.text_edit.insertPlainText(text)
#         self.text_edit.ensureCursorVisible()
#
#     def update_progress_bar(self, value):
#         self.progress_bar.setValue(value)