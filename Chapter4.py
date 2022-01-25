# 생선의 확률을 구한다.
# 먼저 k-nn 회귀를 이용해서 구할것이고 (이웃 분류기기
# 그다음 Logistic Regression을 이용해 구할 것이다.

# 먼저 데이터 가져오기
import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv_data')
# 어떤 특성들이 있을까?
print(pd.unique(fish['Species']))

# 각 특성별로 출력하기(5개의 샘플 생선만)
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print('\n 5 fishes info : ')
print(fish_input[:5])

# 각 원소별로 결과값(target)을 모아보자.
fish_target = fish['Species'].to_numpy()
print('\n fish target : ')
print(fish_target)

# 이제 fish_input과 fish_target으로 train/test로 분류
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state = 42
)

# 전처리 필수
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)



# kn 점수 테스트
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print('\n Train & Test score : ')
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

print('\n KN Classes : ')
print(kn.classes_)

# 처음 5개의 샘플 타깃 출력
print('\n 5 predicts of test_scaled : ')
print(kn.predict(test_scaled[:5]))
# ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch'] 로 예측중. 확률을 구해보자

import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print('\n Each probablity : ')
print(np.round(proba, decimals = 4))
