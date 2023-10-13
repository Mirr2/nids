import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextBrowser, QFileDialog

class DataLoadingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("파일 로더")
        self.setGeometry(100, 100, 600, 400)

        self.load_button = QPushButton("데이터 파일 불러오기", self)
        self.load_button.clicked.connect(self.load_file)
        self.load_button.setGeometry(50, 50, 200, 30)

        self.text = QTextBrowser(self)
        self.text.setGeometry(50, 100, 500, 250)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "데이터 파일 선택")
        if file_path:
            try:
                dataset = pd.read_csv(file_path)
                self.text.append("파일을 성공적으로 불러왔습니다.")
                self.text.append(str(dataset.head()))
            except Exception as e:
                self.text.append(f"파일 불러오기에 실패했습니다: {str(e)}")
        else:
            self.text.append("파일 선택이 취소되었거나 잘못된 파일을 선택하셨습니다.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataLoadingApp()
    window.show()
    sys.exit(app.exec_())