import tkinter as tk
from tkinter import filedialog
import pandas as pd

# 데이터 전처리 함수 (여기에 원하는 전처리 작업을 추가)
def preprocess_data():
    global data
    # 여기에서 데이터 전처리 작업을 수행

# 데이터 불러오기 함수
def load_data():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        preprocess_data()
        show_data()

# 데이터셋 표시 함수
def show_data():
    data_display.delete('1.0', tk.END)  # 이전 데이터 삭제
    data_display.insert('1.0', data)

# GUI 초기화
root = tk.Tk()
root.title("데이터 전처리 및 표시")

# 데이터 불러오기 버튼
load_button = tk.Button(root, text="데이터 불러오기", command=load_data)
load_button.pack()

# 데이터 표시 텍스트 박스
data_display = tk.Text(root)
data_display.pack()

data = None  # 불러온 데이터를 저장할 변수

root.mainloop()
