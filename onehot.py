from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.compose import ColumnTransformer

def onehotencoder(x):
    labelencoder_x_1 = LabelEncoder()
    labelencoder_x_2 = LabelEncoder()

    x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
    x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

    onehotencoder_1 = ColumnTransformer([("tcp", OneHotEncoder(), [1])], remainder='passthrough')
    onehotencoder_2 = ColumnTransformer([("http", OneHotEncoder(), [4])], remainder='passthrough')

    x = np.array(onehotencoder_1.fit_transform(x))
    x = np.array(onehotencoder_2.fit_transform(x))

    return x
