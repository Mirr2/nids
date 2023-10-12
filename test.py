import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton,
                             QFileDialog, QWidget, QLabel, QProgressBar)
from PyQt5.QtCore import QTimer

class DatasetViewer(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.tableWidget = QTableWidget()

        self.loadButton = QPushButton('Load Dataset')
        self.loadButton.clicked.connect(self.loadDataset)

        self.uniqueLabel = QLabel()
        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(100)

        layout.addWidget(self.loadButton)
        layout.addWidget(self.uniqueLabel)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.tableWidget)

        self.setLayout(layout)
        self.show()

    def loadDataset(self):
        filepath, _ = QFileDialog.getOpenFileName()
        if filepath:
            self.progressBar.setValue(0)
            QTimer.singleShot(100, lambda: self.processData(filepath))

    def processData(self, filepath):
        dataset = pd.read_csv(filepath)
        self.progressBar.setValue(30)

        unique_values = dataset['normal.'].unique()
        self.uniqueLabel.setText(f"Unique values before replacement: {unique_values}")
        
        # 카테고리 값 변경
        attack_categories = ['buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.',
                                'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.',
                                'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.',
                                'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
                                'spy.', 'rootkit.']
        dataset['normal.'] = dataset['normal.'].replace(attack_categories, 'attack')
        
        unique_values_after = dataset['normal.'].unique()
        self.uniqueLabel.setText(f"Unique values after replacement: {unique_values_after}")
        self.progressBar.setValue(60)

        self.displayDataset(dataset.head())
        self.progressBar.setValue(100)

    def displayDataset(self, df):
        self.tableWidget.setRowCount(df.shape[0])
        self.tableWidget.setColumnCount(df.shape[1])

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))

app = QApplication(sys.argv)
window = DatasetViewer()
sys.exit(app.exec_())
