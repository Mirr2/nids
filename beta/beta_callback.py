from keras.callbacks import Callback
from PyQt5.QtCore import pyqtSignal, QObject

class CustomCallback(Callback, QObject):
    update_signal = pyqtSignal(str)
    update_progress_signal = pyqtSignal(int)
    
    def __init__(self, text_edit, progress_bar):
        Callback.__init__(self)
        QObject.__init__(self)
        self.text_edit = text_edit
        self.progress_bar = progress_bar
        self.update_signal.connect(self.update_text_edit)
        self.update_progress_signal.connect(self.update_progress_bar)
        
    def on_epoch_end(self, epoch, logs=None):
        text = f"Epoch {epoch+1}/{self.params['epochs']} - " \
               f"loss: {logs['loss']:.4f} - " \
               f"accuracy: {logs['accuracy']:.4f}\n"
        self.update_signal.emit(text)
        # 프로그레스 바 업데이트 (70 + 30 * (현재 에폭 + 1) / 전체 에폭 수)
        progress_value = 70 + 30 * (epoch + 1) / self.params['epochs']
        self.update_progress_signal.emit(int(progress_value))
        
    def on_batch_end(self, batch, logs=None):
        text = f"Batch {batch+1}/{self.params['steps']} - " \
               f"loss: {logs['loss']:.4f} - " \
               f"accuracy: {logs['accuracy']:.4f}\n"
        self.update_signal.emit(text)
        
    def update_text_edit(self, text):
        self.text_edit.insertPlainText(text)
        self.text_edit.ensureCursorVisible()
        
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
