import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from beta_ColumnSelector import ColumnSelector

# DataPreprocessor 클래스 정의
class DataPreprocessor:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.x = None
        self.y = None
        self.encoding_type = None

    def preprocess_data(self):
        self._replace_categories()
        self._split_features_labels()

    def select_column_for_normalization(self, parent=None):
        if 'normal.' in self.dataset.columns:
            return  # 'normal.' 열이 이미 있으면 그냥 반환

        column_selector_dialog = ColumnSelector(self.dataset.columns, parent)
        if column_selector_dialog.exec_():
            selected_column = column_selector_dialog.get_selected_column()
            if selected_column:
                self._rename_column_to_normal(selected_column)

    # 선택된 열의 이름을 'normal.'으로 변경하고 데이터셋의 마지막으로 이동
    def _rename_column_to_normal(self, column_name):
        # 'normal.' 칼럼을 새로 만들고, 선택된 칼럼의 데이터를 복사
        self.dataset['normal.'] = self.dataset[column_name]

        # 원래의 선택된 칼럼을 삭제
        self.dataset.drop(columns=[column_name], inplace=True)

        # 'normal.' 칼럼을 데이터셋의 마지막으로 이동
        normal_column = self.dataset.pop('normal.')
        self.dataset.insert(len(self.dataset.columns), 'normal.', normal_column)

    def _replace_categories(self):
        categories_to_replace = [category for category in self.dataset['normal.'].unique() if category != 'normal.']
        self.dataset['normal.'] = self.dataset['normal.'].replace(categories_to_replace, 'attack')

    def _split_features_labels(self):
        self.x = self.dataset.iloc[:, :self.dataset.shape[1] - 1].values
        self.y = self.dataset.iloc[:, self.dataset.shape[1] - 1].values
        self.y = LabelEncoder().fit_transform(self.y)

    def set_encoding_type(self, encoding_type):
        self.encoding_type = encoding_type

    def label_encode_string_columns(self):
        string_columns_idx = [idx for idx, dtype in enumerate(self.x[0]) if isinstance(dtype, str)]
        label_encoders = [LabelEncoder() for _ in string_columns_idx]

        for i, idx in enumerate(string_columns_idx):
            self.x[:, idx] = label_encoders[i].fit_transform(self.x[:, idx])

    def one_hot_encode_string_columns(self):
        string_columns_idx = [idx for idx, dtype in enumerate(self.x[0]) if isinstance(dtype, str)]

        column_transformers = []
        for i, idx in enumerate(string_columns_idx):
            column_transformers.append(("encoder_" + str(idx), OneHotEncoder(), [idx]))

        ct = ColumnTransformer(column_transformers, remainder='passthrough')
        self.x = ct.fit_transform(self.x)