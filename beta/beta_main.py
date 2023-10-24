import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem, QComboBox, QInputDialog, QLabel,QPushButton,QScrollArea,QVBoxLayout
from PyQt5.uic import loadUi
from beta_data_preprocessor import DataPreprocessor
from beta_model_trainer import ModelTrainer
from beta_model_evaluator import ModelEvaluator
import pandas as pd
import numpy as np

class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi("beta_ui.ui", self)

        self.result_scrollarea = QScrollArea(self)
        self.result_label = QLabel(self.result_scrollarea)

        # 스크롤 영역 설정
        self.result_scrollarea.setWidgetResizable(True)
        self.result_scrollarea.setWidget(self.result_label)

        # 결과를 표시할 위치 설정
        layout = QVBoxLayout()
        layout.addWidget(self.result_scrollarea)

        self.result_label = QLabel(self)
        self.result_scrollarea.setWidget(self.result_label)

        self.data_preprocessor = None
        self.tableWidget = QTableWidget(self)
        self.dataset_scrollarea.setWidget(self.tableWidget)
        self.selected_preprocessing_label = QLabel(self)

        self.tableWidget = QTableWidget(self)
        self.dataset_scrollarea.setWidget(self.tableWidget)
        self.Database_select_button.clicked .connect(self.select_dataset)

        self.Preprocessing_select_button.clicked.connect(self.show_preprocessing_selection)


        # 활성화 함수 선택 콤보박스에 이벤트 연결
        self.hidden_layer_select.currentIndexChanged.connect(self.train_model)

        # 출력 레이어 유형 선택 콤보박스에 이벤트 연결
        self.output_layer_select.currentIndexChanged.connect(self.train_model)

    def select_dataset(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            self.dataset_filepath = filePath
            self.data_preprocessor = DataPreprocessor(filePath)
            self.data_preprocessor.preprocess_data()
            self.dataset = pd.read_csv(filePath)
            self.dataset_filename_label.setText(filePath.split('/')[-1])

    def show_preprocessing_selection(self):
        algorithms = ["one-hot encoding", "label-encoding"]
        selected_algorithm, okPressed = QInputDialog.getItem(self, "Select Algorithm", "Algorithm:", algorithms, 0,
                                                             False)

        if okPressed and selected_algorithm:
            self.selected_preprocessing_label.setText(selected_algorithm)
            result_label_text = f"Selected Algorithm: {selected_algorithm}\n"

            if selected_algorithm == "one-hot encoding":
                if self.data_preprocessor:
                    self.data_preprocessor.one_hot_encode_string_columns()
                    preprocessed_data = self.data_preprocessor.x
                    result_label_text += f"Preprocessed Data:\n"
                    for row in preprocessed_data[:5]:  # 처음 5개 행만 선택하여 출력
                        result_label_text += str(row) + "\n"
            elif selected_algorithm == "label-encoding":
                if self.data_preprocessor:
                    self.data_preprocessor.label_encode_string_columns()
                    preprocessed_data = self.data_preprocessor.x
                    result_label_text += f"Preprocessed Data:\n"
                    for row in preprocessed_data[:5]:  # 처음 5개 행만 선택하여 출력
                        result_label_text += str(row) + "\n"

            # 결과를 보여줄 레이블(Label) 등의 위젯에 설정
            self.result_label.setText(result_label_text)

    def fillTable(self):
        self.tableWidget.setRowCount(self.dataset.shape[0])
        self.tableWidget.setColumnCount(self.dataset.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(self.dataset.columns)

        for i in range(self.dataset.shape[0]):
            for j in range(self.dataset.shape[1]):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.dataset.iat[i, j])))

    def train_model(self):
        selected_activation = self.hidden_layer_select.currentText()
        selected_output_layer = self.output_layer_select.currentText()

        model_trainer = ModelTrainer(self.data_preprocessor.x, self.data_preprocessor.y)
        x_test, y_test = model_trainer.train_model_with_activation(selected_activation, selected_output_layer)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
