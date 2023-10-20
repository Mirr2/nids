import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def onehotencode(df):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, 41].values

    labelencoder_x_1 = LabelEncoder()
    labelencoder_x_2 = LabelEncoder()
    labelencoder_x_3 = LabelEncoder()

    x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
    x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
    x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])

    onehotencoder_1 = ColumnTransformer([("tcp", OneHotEncoder(), [1])], remainder='passthrough')
    onehotencoder_2 = ColumnTransformer([("http", OneHotEncoder(), [4])], remainder='passthrough')
    onehotencoder_3 = ColumnTransformer([("SF", OneHotEncoder(), [70])], remainder='passthrough')

    x = np.array(onehotencoder_1.fit_transform(x))
    x = np.array(onehotencoder_2.fit_transform(x))
    x = np.array(onehotencoder_3.fit_transform(x))

    y = LabelEncoder().fit_transform(y)

    return x, y
