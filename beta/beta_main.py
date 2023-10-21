import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem, QComboBox
from PyQt5.uic import loadUi
import pandas as pd

from beta_data_preprocessor import DataPreprocessor
from beta_model_trainer import ModelTrainer
from beta_model_evaluator import ModelEvaluator

class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi("beta_ui.ui", self)

        self.data_preprocessor = None
        
        self.tableWidget = QTableWidget(self)
        self.dataset_scrollarea.setWidget(self.tableWidget)

        self.Database_select_button.clicked.connect(self.select_dataset)
        self.train_button.clicked.connect(self.train_model)

        # 활성화 함수 선택 콤보박스에 이벤트 연결
        self.hidden_layer_select.currentIndexChanged.connect(self.train_model)

        # 출력 레이어 유형 선택 콤보박스에 이벤트 연결
        self.output_layer_select.currentIndexChanged.connect(self.train_model)





    def select_dataset(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            self.data_preprocessor = DataPreprocessor(filePath)
            self.data_preprocessor.preprocess_data()
            self.dataset = self.data_preprocessor.dataset  # 전처리된 데이터셋
            self.dataset_filename_label.setText(filePath.split('/')[-1])
            self.dataset_path_label.setText(filePath)
            self.fillTable()

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
        self.trained_model, self.x_test, self.y_test = model_trainer.train_model_with_activation(selected_activation, selected_output_layer)

        # 모델 평가
        model_evaluator = ModelEvaluator(self.trained_model)  
        self.accuracy = model_evaluator.evaluate_model(self.x_test, self.y_test)
        self.f1 = model_evaluator.calculate_f1_score(self.x_test, self.y_test)

        self.accuracy_f1score.setText(f"Accuracy: {self.accuracy} / F1 Score: {self.f1}")

        # 결과 출력 (GUI에 표시 또는 콘솔에 출력)
        print(f"Accuracy: {self.accuracy}")
        print(f"F1 Score: {self.f1}")

    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()

