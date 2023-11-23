import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem, QComboBox
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem

import pandas as pd

from beta_data_preprocessor import DataPreprocessor
from beta_model_trainer import ModelTrainer
from beta_model_evaluator import ModelEvaluator

class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi("beta_ui.ui", self)
        # Create QTableWidget for x_test
        self.x_test_table = QTableWidget()
        self.x_test_table.setColumnCount(2)  # Adjust the number of columns based on your x_test data
        self.x_test_table.setHorizontalHeaderLabels(["Feature 1", "Feature 2"])  # Set the headers as per your features
        self.x_test_scrollarea.setWidget(self.x_test_table)


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
        x_test, y_test = model_trainer.train_model_with_activation(selected_activation, selected_output_layer)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
        # Load x_test data into the table
        model_trainer = ModelTrainer()  # Assuming you have an instance of ModelTrainer
        x_test_data = model_trainer.get_x_test()
        self.x_test_table.setRowCount(len(x_test_data))
        for i, row_data in enumerate(x_test_data):
            for j, cell_data in enumerate(row_data):
                self.x_test_table.setItem(i, j, QTableWidgetItem(str(cell_data)))

