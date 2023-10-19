import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QTextEdit, QVBoxLayout, \
    QWidget, QComboBox, QProgressBar, QGraphicsView
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class DataPreprocessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('데이터 전처리 GUI')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.label = QLabel('데이터 전처리를 수행할 파일을 선택하십시오:')
        self.layout.addWidget(self.label)

        self.open_button = QPushButton('파일 열기')
        self.open_button.clicked.connect(self.openFile)
        self.layout.addWidget(self.open_button)

        self.text_edit = QTextEdit()
        self.layout.addWidget(self.text_edit)

        self.process_button = QPushButton('전처리 실행')
        self.process_button.clicked.connect(self.preprocessData)
        self.layout.addWidget(self.process_button)

        self.preprocess_label = QLabel('전처리 방식을 선택하십시오:')
        self.layout.addWidget(self.preprocess_label)

        self.preprocess_combo = QComboBox()
        self.preprocess_combo.addItem('대문자로 변환')
        self.preprocess_combo.addItem('소문자로 변환')
        self.layout.addWidget(self.preprocess_combo)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.canvas = FigureCanvas(plt.figure())  # 그래프를 표시할 캔버스
        self.layout.addWidget(self.canvas)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateProgressBar)

        self.loss = []
        self.accuracy = []

    def openFile(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, '파일 열기', '', '데이터 파일 (*.csv *.txt);;모든 파일 (*)',
                                                   options=options)

        if file_path:
            with open(file_path, 'r') as file:
                data = file.read()
                self.text_edit.setPlainText(data)

    def preprocessData(self):
        data = self.text_edit.toPlainText()
        selected_preprocess = self.preprocess_combo.currentText()

        if selected_preprocess == '대문자로 변환':
            processed_data = data.upper()
        elif selected_preprocess == '소문자로 변환':
            processed_data = data.lower()
        else:
            processed_data = data

        self.text_edit.setPlainText(processed_data)

        # 손실과 정확도 데이터를 가상으로 생성
        self.loss = [0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        self.accuracy = [0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

        # 그래프를 업데이트
        self.updateGraph()

    def updateProgressBar(self):
        current_value = self.progress_bar.value()
        if current_value >= 100:
            self.timer.stop()
        else:
            self.progress_bar.setValue(current_value + 5)  # 5%씩 증가하는 예제

    def updateGraph(self):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.loss, label='손실')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.accuracy, label='정확도')
        plt.legend()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataPreprocessingApp()
    window.show()
    sys.exit(app.exec_())