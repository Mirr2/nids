import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem
from PyQt5.uic import loadUi
import pandas as pd

class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi("testpy.ui", self)

        self.tableWidget = QTableWidget(self)
        self.dataset_scrollarea.setWidget(self.tableWidget)

        # "Database_select_button"이 클릭되면 self.loadCsvFile 함수를 실행
        self.Database_select_button.clicked.connect(self.loadCsvFile)

    def loadCsvFile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        if fname[0]:
            self.dataset = pd.read_csv(fname[0])  # 선택한 CSV 파일을 불러옴
            self.dataset_path_label.setText(fname[0])
            self.dataset_filename_label.setText(os.path.basename(fname[0]))

            # 테이블에 데이터 채우기
            self.fillTable()

    def fillTable(self):
        self.tableWidget.setRowCount(self.dataset.shape[0])
        self.tableWidget.setColumnCount(self.dataset.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(self.dataset.columns)

        for i in range(self.dataset.shape[0]):
            for j in range(self.dataset.shape[1]):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.dataset.iat[i, j])))
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
