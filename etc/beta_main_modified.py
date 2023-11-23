import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem, QComboBox, QProgressBar, QVBoxLayout
from PyQt5.uic import loadUi
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog


from beta_data_preprocessor import DataPreprocessor
from beta_model_trainer import ModelTrainer

    def plot_confusion_matrix(self, y_true, y_pred):
        conf_mat = confusion_matrix(y_true, y_pred)
        ax = self.figure.add_subplot(122)  # Adding a subplot beside the graph
        sns.heatmap(conf_mat, annot=True, ax=ax, fmt='g')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
from beta_model_evaluator import ModelEvaluator


class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi("beta_ui.ui", self)

        self.data_preprocessor = None
        
        self.tableWidget = QTableWidget(self)
        self.dataset_scrollarea.setWidget(self.tableWidget)

        self.fig, self.ax = plt.subplots(2, 1, figsize=(5, 4))  # 2x1 subplots
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.loss_accuracy_graph.setLayout(layout)

        self.Database_select_button.clicked.connect(self.select_dataset)
        self.train_button.clicked.connect(self.train_model)
        self.hidden_layer_select.currentIndexChanged.connect(self.train_model)
        self.output_layer_select.currentIndexChanged.connect(self.train_model)

        self.x_test_table = None
        self.progress_bar = self.findChild(QProgressBar, 'progressBar')
        self.progress_bar.setValue(1)





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
        dialog.setWindowTitle("Accuracy and Loss Graph")

        fig, axs = plt.subplots(2, 1, figsize=(5, 4))
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

    # ... existing code in plot_graph ...
    
    # Plotting the confusion matrix
    self.plot_confusion_matrix(y_true, y_pred)  # y_true and y_pred should be defined or passed

