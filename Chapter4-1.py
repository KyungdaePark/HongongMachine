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
print("train_scaled : " + str(kn.score(train_scaled, train_target)))
print("test_scaled : " + str(kn.score(test_scaled, test_target)))

print('\n KN Classes : ')
print(kn.classes_)

# 처음 5개의 샘플 타깃 출력
print('\n 5 predicts of test_scaled : ')
print(kn.predict(test_scaled[:5]))
# ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch'] 로 예측중. 확률을 구해보자

# 각각의 확률 출력
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print('\n Each probablity : ')
print(np.round(proba, decimals = 4))

# 근데 이거 왜 1/3, 2/3, 3/3 으로만 되는걸까 ?
# 확률처럼 보이게 하는 방법은 ?
# 0부터 1 사이로 보이게 하는 시그모이드 함수를 사용해 보자.
# 1/ 1 + exp(-z) (z 는 확률)
# 시그모이드 함수 그리면 이렇게 됨 : 

import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5,5,0.1) # -5와 5 사이에 0.1 간격으로 계산
phi = 1/(1+np.exp(-z))
plt.plot(z,phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.title("Sigmoid Function")
plt.show()

# 간단하게 시그모이드 함수 꼴로 나오게 , 이진분류를 이용해 보여주겠음
# Boolean indexing을 이용해 도미(Bream) 와 빙어(Smelt)만 골라내겠음
print("\n ### From here, there are only Bream and Smelt fishes ###")
    
bream_smelt_indexes = (train_target == "Bream") | (train_target == "Smelt")
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱 회귀 (Logistic Regression)을 이용해 학습하고 확률을 구할 것.
# 학습하는 방법 : 1. 예측을 해서(fit) 방정식의 계수(coefficient)를 정의한다.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print("\nCoefficients of Logistic Regression for Binary Classification :")
print("coef : " + str(lr.coef_))
print("intercept : " + str(lr.intercept_))

# 2. 정해진 계수모델을 바탕으로 예측한다 : 이 계수를 이용해 각각의 원소들을 계산한 결과값 z는?
decisions = lr.decision_function(train_bream_smelt[:5])
print("Z(decisions) : " + str(decisions))

# 2-1. 이 z를 시그모이드 함수에 넣어 0~1사이의 값을 받는다.
# scipy의 explict 함수를 이용하면 시그모이드 함수 결과값을 얻을 수 있다.
from scipy.special import expit
print("\nResult of sigmoid function with decisions by expit function")
print(expit(decisions))

# decisions는 True(양성 클래스) 와 False(음성 클래스)에 대한 결과값을 양/음수로 계산함.
# expit는 오직 True(양성 클래스)에 대한 결과값을 계산함. 양성일수록 1, 음성일수로 0
# 여기서 양성 클래스는 Smelt

# 이제 다중분류를 해보자
# 다중분류는 LogisticRegression을 이용해 7개의 생선을 분류함.
lr2 = LogisticRegression(C=20, max_iter=1000) # C는 1/alpha 즉 C가 작을수록 규제가 큼, max_iter는 충분히 훈련시키기 위한 반복횟수
lr2.fit(train_scaled, train_target)
print("\n Train & Test score")
print("Train Score : " +  str(lr2.score(train_scaled, train_target)))
print("Test Score : " + str(lr2.score(test_scaled, test_target)))

# 점수가 괜찮으니 처음 5개 샘플에 대한 예측
print("\n Predict for 5 test_scaled")
print(lr2.predict(test_scaled[:5]))

#  5개 샘플에 대한 예측 확률 ?
print("\n Percentage of 5 test_scaled") 
proba = lr2.predict_proba(test_scaled[:5])   
print(np.round(proba, decimals = 3))

# 7개의 정보 : 각각의 원소마다 Z 값을 계산함
# SoftMax 함수를 이용
decisions2 = lr2.decision_function(test_scaled[:5])
print("\n Decisions2 (Z) of 7 Species : ")
print(np.round(decisions2, decimals = 2))

# z값 결과를 softmax에 의뢰
from scipy.special import softmax
print("\n PREDICT OF 5 decisions2 using Softmax")
proba2 = softmax(decisions2, axis = 1) # axis=1이면 각 샘플에 대한 소프트맥스를 계산함.
print(np.round(proba2, decimals = 3))
      
