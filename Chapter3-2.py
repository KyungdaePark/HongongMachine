# K-NN regression이 아닌 linear regression
# 길이 50인 농어의 무게가 잘못 예측되었음.
from sklearn.linear_model import LinearRegression
import numpy as np
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

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state = 42
)
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)

import matplotlib.pyplot as plt

plt.scatter(train_input, train_target)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


distances, indexes = knr.kneighbors([[50]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

plt.scatter(100,1033,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 길이가 50인 농어의 무게 예측 : 
print(knr.predict([[50]]))

# 실제와 차이 多

# Linear Regression 도입
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(train_input, train_target)
print(lr.predict([[50]]))

# length = 15 ~ 50까지 그래프를 그리면
import matplotlib.pyplot as plt
plt.scatter(train_input, train_target)
plt.plot([15,50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_+lr.intercept_])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 하지만 y절편이 음수이며 score을 확인해보면 과소적합이 일어남
# 그래서 다항회귀 사용

train_poly = np.column_stack((train_input ** 2 , train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
# y = ax^2 + bx + c 에서 x = input이 되는것.
# 기존의 input은 [42,1] shape 이지만 여기서는 [42,2]가 됨.
# broadcasting 으로 자동으로 numpy에서 곱해줌

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50 ** 2, 50]])) # 50을 넣었을 때 다항회귀 결과값 확인
# train 할때에도 [x^2, x]로 넣어줬으므로 동일하게 넣어줘야 함.
# = 1573.98423528

# 그래프 그리기 : 15부터 50까지 길이가 커지면 어떻게 변할까

point = np.arange(15,50)
plt.scatter(train_input, train_target)
plt.plot(point, lr.coef_[0]*point**2 + lr.coef_[1]*point +  lr.intercept_)

plt.scatter(50,lr.predict([[50 **2, 50]]), marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))