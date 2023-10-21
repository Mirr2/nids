import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton, QWidget, QGridLayout
from PyQt5.uic import loadUi
import pandas as pd

class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi("tabletestui.ui", self)

        #1 첫 번째 테이블 위젯
        self.tableWidget = QTableWidget(self)
        self.dataset_scrollarea.setWidget(self.tableWidget)

        # 두 번째 테이블 위젯
        self.xTestTableWidget = QTableWidget(self)
        self.x_test_scrollarea.setWidget(self.xTestTableWidget)

        # 세 번째 테이블 위젯
        self.xTestResultTableWidget = QTableWidget(self)
        self.x_test_scrollarea_2.setWidget(self.xTestResultTableWidget)

        # "Database_select_button"이 클릭되면 self.loadCsvFile 함수를 실행
        self.Database_select_button.clicked.connect(self.loadCsvFile)

        # 첫 번째 테이블 셀 클릭 이벤트를 연결
        self.tableWidget.cellClicked.connect(self.showXTest)

        # 두 번째 테이블 셀 클릭 이벤트를 연결
        self.xTestTableWidget.cellClicked.connect(self.showXTestResult)

        self.selected_dataset = None  # 선택된 데이터셋을 저장할 변수
        self.selected_x_test = None  # 선택된 x_test 값을 저장할 변수


    def loadCsvFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        if fname[0]:
            self.dataset = pd.read_csv(fname[0])  # 선택한 CSV 파일을 불러옴
            self.dataset_path_label.setText(fname[0])
            self.dataset_filename_label.setText(os.path.basename(fname[0]))

            self.selected_dataset = self.dataset  # 선택된 데이터셋을 저장

            self.fillTable()

    def fillTable(self):
        self.tableWidget.setRowCount(self.dataset.shape[0])
        self.tableWidget.setColumnCount(self.dataset.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(self.dataset.columns)

        for i in range(self.dataset.shape[0]):
            for j in range(self.dataset.shape[1]):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.dataset.iat[i, j])))

    def showXTest(self, row, column):
        if self.selected_dataset is not None:
            # 선택한 데이터셋에서 선택한 행의 데이터를 가져와서 xTestTableWidget에 표시
            selected_data = self.selected_dataset.iloc[row]
            x_test = selected_data.to_frame().T  # 행 데이터를 열로 변환
            self.selected_x_test = x_test
            self.fillXTestTable(self.xTestTableWidget, x_test)

    def showXTestResult(self, row, column):
        if self.selected_x_test is not None:
            # 선택한 x_test에서 선택한 행의 데이터를 가져와서 xTestResultTableWidget에 표시
            selected_data = self.selected_x_test.iloc[:, column]
            self.fillXTestTable(self.xTestResultTableWidget, selected_data)

    def fillXTestTable(self, table_widget, data):
        table_widget.clear()
        table_widget.setRowCount(data.shape[0])
        table_widget.setColumnCount(data.shape[1])
        table_widget.setHorizontalHeaderLabels(data.columns)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iat[i, j]))
                table_widget.setItem(i, j, item)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()