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
        self.x_test_tableWidget = self.x_test_scrollarea.findChild(QTableWidget)


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
        x_test, y_test = model_trainer.train_model_with_activation(selected_activation, selected_output_layer)
        self.show_x_test_data(x_test)
    
    def show_x_test_data(self, x_test):
        self.x_test_tableWidget.setRowCount(x_test.shape[0])
        self.x_test_tableWidget.setColumnCount(x_test.shape[1])
        self.x_test_tableWidget.setHorizontalHeaderLabels([str(i) for i in range(x_test.shape[1])])

        for i in range(x_test.shape[0]):
            for j in range(x_test.shape[1]):
                self.x_test_tableWidget.setItem(i, j, QTableWidgetItem(str(x_test[i, j])))

    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
