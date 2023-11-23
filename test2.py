from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
import sys

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('testpy.ui', self)
        
        self.data_preprocessor = None
        self.model_trainer = None
        self.x_test = None
        self.y_test = None

        self.Database_select_button.clicked.connect(self.select_dataset)
        self.Preprocessing_select_button.clicked.connect(self.select_preprocessing)
        self.Algorithm_select_button.clicked.connect(self.select_algorithm)
        self.hidden_layer_select.clicked.connect(self.select_hidden_layer)
        self.output_layer_select.clicked.connect(self.select_output_layer)
        self.pushButton.clicked.connect(self.run_model)

    def select_dataset(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filePath:
            self.data_preprocessor = DataPreprocessor(filePath)
            self.dataset_filename_label.setText(filePath.split('/')[-1])
            self.dataset_path_label.setText(filePath)

    def select_preprocessing(self):
        if self.data_preprocessor:
            self.data_preprocessor.preprocess_data()

    def select_algorithm(self):
        if self.data_preprocessor:
            self.model_trainer = ModelTrainer(self.data_preprocessor.x, self.data_preprocessor.y)

    def select_hidden_layer(self):
        pass  # Implement hidden layer selection logic here

    def select_output_layer(self):
        pass  # Implement output layer selection logic here

    def run_model(self):
        if self.model_trainer:
            self.x_test, self.y_test = self.model_trainer.train_model()
            model_evaluator = ModelEvaluator(self.model_trainer.model)
            accuracy = model_evaluator.evaluate_model(self.x_test, self.y_test)
            self.accuracy_f1score.setText(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
