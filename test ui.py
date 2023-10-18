import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QTextEdit, QFileDialog, QInputDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextOption
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 머신러닝 알고리즘을 선택할 변수
selected_algorithm = ""
selected_encoding = ""

class DataPreprocessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('데이터 전처리')
        self.setGeometry(100, 100, 400, 300)

        # 파일 선택 버튼
        self.file_select_button = QPushButton('파일 선택', self)
        self.file_select_button.clicked.connect(self.select_file)

        # 데이터 전처리 버튼
        self.preprocess_button = QPushButton('데이터 전처리', self)
        self.preprocess_button.clicked.connect(self.preprocess_data)
        self.preprocess_button.setDisabled(True)

        # 순위 보기 버튼
        self.view_ranking_button = QPushButton('순위 보기', self)
        self.view_ranking_button.clicked.connect(self.view_ranking)
        self.view_ranking_button.setDisabled(True)

        # 알고리즘 선택 버튼
        self.choose_algorithm_button = QPushButton('알고리즘 선택', self)
        self.choose_algorithm_button.clicked.connect(self.choose_algorithm)

        # 원핫 인코딩 적용 버튼
        self.onehot_encoding_button = QPushButton('One-Hot encoding', self)
        self.onehot_encoding_button.clicked.connect(self.apply_onehot_encoding)
        self.onehot_encoding_button.setDisabled(True)

        # 특징 및 레이블 선택 버튼
        self.select_features_and_labels_button = QPushButton('특징 및 레이블 선택', self)
        self.select_features_and_labels_button.clicked.connect(self.select_features_and_labels)
        self.select_features_and_labels_button.setDisabled(True)

        # 결과 텍스트 레이블
        self.result_label = QTextEdit(self)
        self.result_label.setReadOnly(True)
        self.result_label.setWordWrapMode(QTextOption.NoWrap)

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.file_select_button)
        layout.addWidget(self.preprocess_button)
        layout.addWidget(self.onehot_encoding_button)
        layout.addWidget(self.select_features_and_labels_button)
        layout.addWidget(self.view_ranking_button)
        layout.addWidget(self.choose_algorithm_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", "모든 파일 (*)", options=options)
        if file_name:
            self.data_file = file_name
            self.preprocess_button.setEnabled(True)
            self.onehot_encoding_button.setDisabled(True)
            self.select_features_and_labels_button.setDisabled(True)
            print(f"선택한 파일: {file_name}")

    def preprocess_data(self):
        data = pd.read_csv(self.data_file)

        x = data.iloc[:, :-1].values
        y = data.iloc[:, 41].values

        self.onehot_encoding_button.setEnabled(True)
        self.select_features_and_labels_button.setEnabled(True)

    def apply_onehot_encoding(self):
        global selected_encoding
        selected_encoding = "onehot"

        data = pd.read_csv(self.data_file)
        x = data.iloc[:, :-1].values
        y = data.iloc[:, 41].values

        labelencoder_x_1 = LabelEncoder()
        labelencoder_x_2 = LabelEncoder()
        labelencoder_x_3 = LabelEncoder()

        x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
        x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
        x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])

        onehotencoder_1 = ColumnTransformer([("tcp", OneHotEncoder(), [1])], remainder='passthrough')
        onehotencoder_2 = ColumnTransformer([("http", OneHotEncoder(), [2])], remainder='passthrough')
        onehotencoder_3 = ColumnTransformer([("SF", OneHotEncoder(), [3])], remainder='passthrough')

        x_encoded = onehotencoder_1.fit_transform(x)
        x_encoded = onehotencoder_2.fit_transform(x_encoded)
        x_encoded = onehotencoder_3.fit_transform(x_encoded)

        labelencoder_y = LabelEncoder()
        y_encoded = labelencoder_y.fit_transform(y)

        result = "데이터 전처리 완료 (원핫 인코딩 적용)\n"
        result += f"X shape: {x_encoded.shape}\n"
        result += f"Y shape: {y_encoded.shape}\n"
        result += f"Number of columns: {x_encoded.shape[1]}\n"

        self.result_label.setPlainText(result)

    def select_features_and_labels(self):
        global selected_encoding
        selected_encoding = "custom"

        data = pd.read_csv(self.data_file)
        x = data.iloc[:, :-1].values
        y = data.iloc[:, 41].values

        # 사용자 지정 전처리 방식을 구현
        # x와 y를 적절하게 처리

        result = "데이터 전처리 완료 (사용자 지정 방식)\n"
        result += f"X shape: {x.shape}\n"
        result += f"Y shape: {y.shape}\n"
        result += f"Number of columns: {x.shape[1]}\n"

        self.result_label.setPlainText(result)

    def view_ranking(self):
        rankings = ["순위 1", "순위 2", "순위 3", "순위 4", "순위 5"]
        rankings.sort()
        rankings_str = "\n".join(rankings)
        self.result_label.setPlainText(rankings_str)

    def choose_algorithm(self):
        global selected_algorithm
        items = ("머신러닝", "딥러닝")
        item, okPressed = QInputDialog.getItem(self, "알고리즘 선택", "알고리즘을 선택하세요:", items, 0, False)
        if okPressed and item:
            selected_algorithm = item
            print(f"선택한 알고리즘: {selected_algorithm}")

def main():
    app = QApplication(sys.argv)
    window = DataPreprocessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()