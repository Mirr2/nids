import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# DataPreprocessor 클래스 정의
class DataPreprocessor:
    def __init__(self,dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.x = None
        self.y = None

    def preprocess_data(self):
        self._replace_categories()
        self._split_features_labels()

    def _replace_categories(self):
        categories_to_replace = [category for category in self.dataset['normal.'].unique() if category != 'normal.']
        self.dataset['normal.'] = self.dataset['normal.'].replace(categories_to_replace, 'attack')

    def _split_features_labels(self):
        self.x = self.dataset.iloc[:, :self.dataset.shape[1]-1].values
        self.y = self.dataset.iloc[:, self.dataset.shape[1]-1].values

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
