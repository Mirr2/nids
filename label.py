import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('test.csv')

def label_encode(dataset):
    # 'normal.' 레이블을 제외한 모든 레이블을 'attack'으로 변경
    dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.',
                                                     'guess_passwd.', 'imap.', 'ipsweep.',
                                                     'land.', 'loadmodule.', 'multihop.',
                                                     'neptune.', 'nmap.',
                                                     'perl.', 'phf.', 'pod.', 'portsweep.',
                                                     'rootkit.', 'satan.', 'smurf.',
                                                     'spy.', 'teardrop.', 'warezclient.',
                                                     'warezmaster.'],
                                                    "attack")

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 41].values

    # Define the order of labels for y and encode it.
    unique_labels = dataset['normal.'].unique()
    le_y = LabelEncoder()
    le_y.classes_ = unique_labels
    y = le_y.transform(y)

    return x, y

# Apply label encoding
x, y = label_encode(df)

