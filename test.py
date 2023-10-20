import pandas as pd
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox,QLabel

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('testpy.ui', self)

        self.file_path = None

        # Button 클릭 이벤트 연결
        self.Database_select_button.clicked.connect(self.select_database)

        self.selected_preprocessing_label = QtWidgets.QLabel(self)  # QLabel을 생성
        self.selected_preprocessing_label.setGeometry(170, 70, 121, 21)  # 위치 및 크기 설정
        self.selected_preprocessing_label.hide()  # 일단 숨김

        # Preprocessing_select_button 버튼 클릭 시 알고리즘 선택 창과 레이블 표시 이벤트 연결
        self.Preprocessing_select_button.clicked.connect(self.show_preprocessing_selection)

        self.Algorithm_select_button.clicked.connect(self.show_Algorithm_selection)

    def select_database(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(filter="CSV Files (*.csv)")

        if file_path:
            print(f"Selected file: {file_path}")

            file_name = file_path.split("/")[-1]

            # Save the selected file path
            self.dataset_filename_label.setText(file_name)

            self.file_path = file_path

    def show_preprocessing_selection(self):
        if self.file_path is None:
            QMessageBox.warning(self, "경고", "파일을 선택해주세요.")
            return

        algorithms = ["one-hot encoding", "label-encoding"]
        selected_algorithm, okPressed = QInputDialog.getItem(self, "Select Algorithm", "Algorithm:", algorithms, 0,
                                                             False)

        if okPressed:
            # Update the label with the selected preprocessing's name.
            self.selected_preprocessing_label.setText(selected_algorithm)
            self.selected_preprocessing_label.show()  # 레이블을 보이도록 변경

            df = pd.read_csv(self.file_path)

    def show_Algorithm_selection(self):
        if self.file_path is None:
            QMessageBox.warning(self, "경고", "파일을 선택해주세요.")
            return

        algorithms = ["머신러닝", "딥 러닝"]  # 여기에 원하는 알고리즘 목록 추가
        selected_algorithm, okPressed = QInputDialog.getItem(self, "Select Algorithm", "Algorithm:", algorithms, 0, False)

        if okPressed:
            if selected_algorithm == "머신러닝":
                # 머신러닝 실행 로직을 추가하거나 호출
                pass
            elif selected_algorithm == "딥 러닝":
                # 딥 러닝 실행 코드를 추가
                deep_learning_script = "softmax.py"  # softmax.py 파일 경로

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
