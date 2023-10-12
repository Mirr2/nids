import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score # 정밀도, 재현율

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem

# ====================================================================================
dataset_addr = './dataset/kddcup.data_10_percent_corrected.csv'
# 데이터 셋 선택후 읽어오기
dataset = pd.read_csv(dataset_addr)
print(dataset.head())
# ====================================================================================

# 마지막 label 컬럼 안에 담긴 고유의 카테고리 값을 확인
print(dataset['normal.'].unique())
print("label컬럼 안의 카테고리 값 확인 : "+dataset['normal.'].unique())
# normal. 을 제외한 모든 카테고리를 attack 으로 변경
dataset['normal.'] = dataset['normal.'].replace(['buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.',
       'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.',
       'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.',
       'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
       'spy.', 'rootkit.'], 'attack')
print("변한 카테고리 값 확인 : "+dataset['normal.'].unique())
# ====================================================================================

# x = 마지막 label 열을 제외한 모든 열을 특징으로 사용
# y = 마지막 label 열을 카테고리로 사용
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,41].values

print("xshpae, yshape : "+x.shape, y.shape)
# ====================================================================================

# 1개의 컬럼을 숫자로 바꾸면서 012 이런식으로 칸을 바꿈.
uniq1 = dataset.tcp.unique()
uniq2 = dataset.http.unique()
uniq3 = dataset.SF.unique()

print(uniq1, '\n', uniq2, '\n', uniq3)
print(uniq1.size, '\n', uniq2.size, '\n', uniq3.size)
# ====================================================================================

# tcp(프로토콜), http(서비스), SF(플래그) 열을 One-Hot Encoding하는 것.
from sklearn.compose import ColumnTransformer # 열을 변환시켜주는 역할.

labelencoder_x_1 = LabelEncoder() # labelencoder란? sklearn.preprocessing에서 선언했었음.
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()

x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])
x[:,3] = labelencoder_x_3.fit_transform(x[:,3])

#onehotencoder_1 = OneHotEncoder(categorical_features=[1])
onehotencoder_1= ColumnTransformer([("tcp", OneHotEncoder(), [1])], remainder = 'passthrough')
#onehotencoder_2 = OneHotEncoder(categorical_features=[4])
onehotencoder_2= ColumnTransformer([("http", OneHotEncoder(), [4])], remainder = 'passthrough') # tcp가 123 했으니까 다음 건 4가 들어가야겠지. http 컬럼에 대해선.
#onehotencoder_3 = OneHotEncoder(categorical_features=[70])
onehotencoder_3= ColumnTransformer([("SF", OneHotEncoder(), [70])], remainder = 'passthrough') # 얜 왜 70인가. 4+66 했으니까 70이겠지.

x = np.array(onehotencoder_1.fit_transform(x))
x = np.array(onehotencoder_2.fit_transform(x))
x = np.array(onehotencoder_3.fit_transform(x))

labelencoder_y = LabelEncoder() # 문자를 숫자로 치환
y = labelencoder_y.fit_transform(y)
print(y)
print(x.shape, y.shape)

# 최종 데이터 개수
# x = 494,020 x 118
# y = 494,920 x 1
# ====================================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# x와 y를 훈련 데이터(train)와 테스트 데이터로 분류 = 70 : 30
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 0)

# 가우시안 기반 나이브 베이즈 알고리즘 초기화 및 학습 (fit)
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# 학습한 모델에 테스트 데이터를 넣어 결과를 확인(predict)
y_pred = classifier.predict(x_test)
print(y_pred.shape)
print(y_test.shape)
print(y_pred)
print(y_test)

class TableWindow(QMainWindow):
    def __init__(self, df):
        super().__init__()

        self.table = QTableWidget(self)
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df.index))

        for i, col in enumerate(df.columns):
            for j, val in enumerate(df[col]):
                self.table.setItem(j, i, QTableWidgetItem(str(val)))

        self.table.setHorizontalHeaderLabels(df.columns)
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.show()

dataset = pd.read_csv(dataset_addr)

app = QApplication(sys.argv)
window = TableWindow(dataset.head())
sys.exit(app.exec_())