import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem, QComboBox, QProgressBar, QVBoxLayout, QInputDialog
from PyQt5.uic import loadUi
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog
from sklearn.metrics import confusion_matrix
import seaborn as sns  # seaborn 라이브러리를 사용하여 혼동 행렬을 그립니다.


from beta_data_preprocessor import DataPreprocessor
from beta_model_trainer import ModelTrainer
from beta_model_evaluator import ModelEvaluator


class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi("beta_ui.ui", self)

        self.Preprocessing_select_button.clicked.connect(self.show_preprocessing_selection)

        self.data_preprocessor = None
        
        self.tableWidget = QTableWidget(self)
        self.dataset_scrollarea.setWidget(self.tableWidget)

        self.fig, self.ax = plt.subplots(2, 1, figsize=(5, 4))  # 2x1 subplots
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        self.Database_select_button.clicked.connect(self.select_dataset)
        self.train_button.clicked.connect(self.train_model)
        self.hidden_layer_select.currentIndexChanged.connect(self.train_model)
        self.output_layer_select.currentIndexChanged.connect(self.train_model)

        self.x_test_table = None
        self.progress_bar = self.findChild(QProgressBar, 'progressBar')
        self.progress_bar.setValue(1)



    def show_preprocessing_selection(self):
        algorithms = ["one-hot encoding", "label-encoding"]
        selected_algorithm, okPressed = QInputDialog.getItem(self, "Select Algorithm", "Algorithm:", algorithms, 0, False)

        if okPressed and selected_algorithm:
            self.selected_preprocessing_label.setText(selected_algorithm)
            if selected_algorithm == "one-hot encoding":
                if self.data_preprocessor:
                    self.data_preprocessor.one_hot_encode_string_columns()
            elif selected_algorithm == "label-encoding":
                if self.data_preprocessor:
                    self.data_preprocessor.label_encode_string_columns()




    def select_dataset(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            self.progress_bar.setValue(10)  # 파일 선택 완료, 10%로 설정
            self.data_preprocessor = DataPreprocessor(filePath)
            self.progress_bar.setValue(20)  # 데이터 전처리 시작, 20%로 설정
            self.data_preprocessor.preprocess_data()
            self.progress_bar.setValue(30)  # 데이터 전처리 완료, 30%로 설정
            self.dataset = self.data_preprocessor.dataset  # 전처리된 데이터셋
            self.dataset_filename_label.setText(filePath.split('/')[-1])
            self.dataset_path_label.setText(filePath)
            self.fillTable()
            self.show_preprocessing_selection()


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

        self.progress_bar.setValue(40)  # 전처리 완료
        model_trainer = ModelTrainer(self.data_preprocessor.x, self.data_preprocessor.y)
        self.trained_model, self.x_test, self.y_test, self.accuracy_data, self.loss_data = model_trainer.train_model_with_activation(selected_activation, selected_output_layer)
        self.progress_bar.setValue(70)  # 모델 훈련 완료


        # 모델 평가
        model_evaluator = ModelEvaluator(self.trained_model)  
        self.accuracy = model_evaluator.evaluate_model(self.x_test, self.y_test)
        self.f1 = model_evaluator.calculate_f1_score(self.x_test, self.y_test)
        self.progress_bar.setValue(85)  # 모델 평가 완료

        self.accuracy_f1score.setText(f"Accuracy: {self.accuracy} / F1 Score: {self.f1}")

        # 결과 출력 (GUI에 표시 또는 콘솔에 출력)
        print(f"Accuracy: {self.accuracy}")
        print(f"F1 Score: {self.f1}")
        self.fillXTestTable()
        self.progress_bar.setValue(100)  # 모든 과정 완료

        self.showGraphDialog()

    def showGraphDialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Accuracy, Loss, and Confusion Matrix")

        fig, axs = plt.subplots(3, 1, figsize=(5, 8))  # 3개의 subplot을 가진다.
        canvas = FigureCanvas(fig)

        layout = QVBoxLayout()
        layout.addWidget(canvas)
        dialog.setLayout(layout)

        # 실제 데이터로 그래프 그리기
        epochs = range(1, len(self.accuracy_data) + 1)

        axs[0].plot(epochs, self.accuracy_data)
        axs[0].set_title('Accuracy')

        axs[1].plot(epochs, self.loss_data)
        axs[1].set_title('Loss')

        # 혼동 행렬 계산 및 그리기
        y_pred = self.trained_model.predict(self.x_test).argmax(axis=1)  # 모델의 예측값을 가져옵니다.

        # y_test의 차원에 따라 적절한 axis 값을 설정
        if len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1:
            y_true = self.y_test.argmax(axis=1)  # 실제 레이블을 가져옵니다.
        else:
            y_true = self.y_test  # 이미 1차원 배열이라면 그대로 사용

        # 이제 여기에서 labels를 설정합니다.
        if len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1:
            labels = list(range(self.y_test.shape[1]))  # 다중 레이블의 경우
        else:
            labels = list(set(y_true))  # 단일 레이블의 경우

        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axs[2], xticklabels=labels, yticklabels=labels)
        axs[2].set_title('Confusion Matrix')
        axs[2].set_xlabel('Predicted')
        axs[2].set_ylabel('True')

        dialog.exec_()


    def fillXTestTable(self):
        x_test_df = pd.DataFrame(self.x_test)  # x_test를 DataFrame으로 변환

        # QTableWidget 생성
        self.x_test_table = QTableWidget()
        self.x_test_table.setRowCount(x_test_df.shape[0])
        self.x_test_table.setColumnCount(x_test_df.shape[1])

        # 데이터 채우기
        for i in range(x_test_df.shape[0]):
            for j in range(x_test_df.shape[1]):
                self.x_test_table.setItem(i, j, QTableWidgetItem(str(x_test_df.iat[i, j])))

        # QScrollArea에 QTableWidget 설정
        self.x_test_scrollarea.setWidget(self.x_test_table)

        # x_test 테이블의 아이템 선택 상태가 변경될 때 실행할 메서드 연결
        self.x_test_table.itemSelectionChanged.connect(self.showSelectedRow)
        
        # QScrollArea에 QTableWidget 설정
        self.x_test_scrollarea.setWidget(self.x_test_table)

    def showSelectedRow(self):
        selected_row = self.x_test_table.currentRow()
        selected_data = [self.x_test_table.item(selected_row, col).text() for col in range(self.x_test_table.columnCount())]

        # 선택된 행에 대응하는 y_test 값 가져오기
        corresponding_y_test = self.y_test[selected_row]

        # 콘솔에 출력
        print(f"Corresponding y_test value: {corresponding_y_test}")

        # 선택된 행에 대응하는 y_test 값 가져오기
        corresponding_y_test = self.y_test[selected_row]


        # y_test QTextBrowser에 표시
        self.y_test_textBrowser.setText(str(corresponding_y_test))

        
        # 선택된 행을 새로운 QTableWidget에 표시
        selected_row_table = QTableWidget()
        selected_row_table.setRowCount(1)
        selected_row_table.setColumnCount(len(selected_data))
        
        for j, data in enumerate(selected_data):
            selected_row_table.setItem(0, j, QTableWidgetItem(str(data)))

        # QScrollArea에 새로운 QTableWidget 설정
        self.x_test_scrollarea_2.setWidget(selected_row_table)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()

