import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

plt.scatter(perch_length, perch_weight)
plt.xlabel('lenght')
plt.ylabel('weight')
plt.show()


train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state =42
)
print(train_input)
print(train_target)
# 2차원 배열로 변환, (42,1) 모양이지만 만약 (-1,2)로 reshape 하면 (21,2)가 됨.
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

# 회귀, regression
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))

# 결과가 1이 나오지 않음. 이 knr.score는 R^2, 결정계수(coefficient of determination)
# R^2 = 1 - { SUM(target - predict)^2 / SUM(target-mean)^2}
# 만약 예측이 타깃에 가까워지면 분자가 0이 되어 R^2이 1과 가까워지고,
# 만약 예측이 평균에 가까워지면 분자와 분모가 서로 가까워져 0과 가까워진다.

# 타깃과 예측의 절댓값 오차를 평균내는 mean_absoulte_error
# mean_absolute_error 함수는 타깃과 예측을 뺀 값을 제곱한 다음 전체 샘플에 대해 평균한 값을 반환함.
from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
# 예측이 평균적으로 mae 만큼 차이난다.

# print(knr.score(test_input, test_target)) 는 테스트 결과의 점수였음
# 훈련 세트의 knr 점수는 ?
print(knr.score(train_input, train_target))

# 훈련 세트의 모델이 단순하여 테스트 점수보다 낮음. 이를 과소적합(underfitting) 이라고 함.
# 그 반대는 과대적합(overfitting) 이라고 함.

# 과소적합을 해결하기 위해서 knr의 이웃 수 k를 3으로 줄이고 다시 훈련함.

knr.n_neighbors = 3
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))
print(knr.score(train_input, train_target))


#### P.128 #2 농어의 길이가 [5,45]로 변하고, 이웃 수 k의 값이 1,5,10 이라면?
print(test_input)
knr2 = KNeighborsRegressor()
x = np.arange(5,45).reshape(-1,1)

for n in [1,5,10]:
    knr2.n_neighbors = n
    knr2.fit(train_input, train_target)
    prediction = knr2.predict(x)
    
    plt.scatter(train_input, train_target)
    plt.plot(x,prediction)
    plt.title('n_neighbors = {}'.format(n))
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
    
# n이 커짐에 따라 모델이 단순해진다.