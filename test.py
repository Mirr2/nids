from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QLabel, QInputDialog,QMessageBox,QPlainTextEdit
import numpy as np
from onehot import onehotencoder
import pandas as pd


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('testpy.ui', self)

        self.file_path = None

        # Button 클릭 이벤트 연결
        self.Database_select_button.clicked.connect(self.select_database)

        self.algorithm_label = QLabel('Algorithm selected', self)
        self.algorithm_label.setGeometry(170, 70, 121, 21)
        self.algorithm_label.hide()

        # Preprocessing_select_button 버튼 클릭 시 알고리즘 선택 창과 레이블 표시 이벤트 연결
        self.Preprocessing_select_button.clicked.connect(self.show_preprocessing_selection)

        self.Algorithm_select_button.clicked.connect(self.show_Algorithm_selection)

    def select_database(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(filter="CSV Files (*.csv)")

        if file_path:
            print(f"Selected file: {file_path}")

            file_name = file_path.split("/")[-1]

            # Save the selected file patha
            self.dataset_filename_label.setText(file_name)

            self.file_path = file_path

    def show_preprocessing_selection(self):
        if self.file_path is None:
            QMessageBox.warning(self, "경고", "파일을 선택해주세요.")
            return
        algorithms = ["one-hot encoding", "label"]
        selected_algorithm, okPressed = QInputDialog.getItem(self, "Select Algorithm", "Algorithm:", algorithms, 0,
                                                             False)

        if okPressed and selected_algorithm:
            if selected_algorithm == "one-hot encoding":
                # Load data from the selected CSV file
                df = pd.read_csv(self.file_path)

                # Convert the DataFrame to a numpy array for preprocessing
                x = df.to_numpy()

                # Call the preprocessing function
                x = onehotencoder(x)


            elif selected_algorithm == "label":
                df = pd.read_csv(self.file_path)

                x = df.iloc[:, :-1].values

                y = df.iloc[:, 41].values

        self.algorithm_label.setText(selected_algorithm)
        self.algorithm_label.show()

    def show_Algorithm_selection(self):
        if self.file_path is None:
            QMessageBox.warning(self, "경고", "파일을 선택해주세요.")
            return
        algorithms = ["머신러닝", "딥 러닝"]  # 여기에 원하는 알고리즘 목록 추가
        selected_algorithm, okPressed = QInputDialog.getItem(self,"Select Algorithm","Algorithm:",algorithms,0,False)



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())