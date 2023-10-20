from sklearn.preprocessing import LabelEncoder

# 레이블 인코딩 함수 정의
def label_encoder(labels):
    labelencoder = LabelEncoder()
    encoded_labels = labelencoder.fit_transform(labels)
    return encoded_labels

# 입력 데이터와 해당하는 레이블로 구성된 전체 데이터셋
data = ... # 전체 데이터셋 로드

# 입력 특성과 레이블로 분리
X = data[:, :-1] # 마지막 열 제외한 모든 열은 입력 특성으로 사용됨
y = data[:, -1] # 마지막 열은 해당하는 레이블 값

# 레이블 인코딩 적용
encoded_labels = label_encoder(y)

# 인코딩된 레이블 출력 (옵션)
print(encoded_labels)
