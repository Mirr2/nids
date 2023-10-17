import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np

# 데이터 전처리 함수 (테스트용)
def preprocess_data():
    global data
    # 임의의 데이터 생성 (테스트용)
    data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': np.random.randint(18, 60, 4),
        'Salary': np.random.randint(30000, 90000, 4)
    })

# 데이터셋 표시 함수
def show_data():
    for i, col in enumerate(data.columns):
        tree.column(i, width=100)
        tree.heading(i, text=col)
    for index, row in data.iterrows():
        tree.insert('', 'end', values=row)

# GUI 초기화
root = tk.Tk()
root.title("데이터 전처리 및 표시")

# 데이터 전처리 버튼
preprocess_button = tk.Button(root, text="데이터 전처리", command=preprocess_data)
preprocess_button.pack()

# 표 테이블 (Treeview)
tree = ttk.Treeview(root, columns=data.columns, show='headings')
tree.pack()

# 데이터 표시 버튼
show_button = tk.Button(root, text="데이터 표시", command=show_data)
show_button.pack()

data = None  # 생성된 데이터를 저장할 변수

root.mainloop()
