# 훈련 세트의 점수가 테스트 세트의 점수보다 낮은 것이 문제 (과소적합)
# 가로,세로,두께를 이용해서 훈련하기 : 특성공학(feature engineering)

# perch의 가로 세로 두께 데이터를 가져옴
from math import degrees
import pandas as pd
df = pd.read_csv("http://bit.ly/perch_csv_data")
perch_full = df.to_numpy()
#print(perch_full)

import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state = 42
)

# 사이킷런의 변환기 : 데이터를 이용해 특성공학 시작
# 가로, 세로, 두께 정보를 같은 방식으로 가공하여 훈련함
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

# 어떤 조합으로 만들어졌는지 확인하기
print(poly.get_feature_names_out())

# 이제 poly를 이용해 훈련해보자 : 1. Linear
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)

print("\nPoly -> Linear score : ")
print("train score : " + str(lr.score(train_poly, train_target)))
print("test score : " + str(lr.score(test_poly, test_target)))


# 곱하는 계수를 증가시켜 더 정확한 모델을 구현할 수 있음 : degree
poly2 = PolynomialFeatures(degree = 5, include_bias = False)
poly2.fit(train_input)
train_poly2 = poly2.transform(train_input)
test_poly2 = poly2.transform(test_input)

lr.fit(train_poly2, train_target)
print("\nPoly2 -> Linear score : ")
print("train score : " + str(lr.score(train_poly2, train_target)))
print("test score : " + str(lr.score(test_poly2, test_target)))

# train score는 매우 높게 나오지만 test는 너무 낮음.
# 과대적합 (overfitting), 특성이 너무 많아서(55개) 문제임
# 이럴때 사용하는게 규제(regulariztion)

# 규제는 계수를 제어하는건데, 애초에 게수의 스케일 자체가 다르다면 공정하게 제어되지 않음
# 그래서 정규화를 해주어야함 (preprocessing)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly2)
train_scaled = ss.transform(train_poly2)
test_scaled = ss.transform(test_poly2)

# 규제에는 두가지 방법 : 릿지(ridge)와 라쏘(lasso)
# ridge : 계수를 제곱한 값을 기준으로 규제를 정함
# lasso : 게수의 절댓값을 기준으로 규제를 정함

from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print("\nPoly2 -> Ridge -> Linear score : ")
print("train score : " + str(ridge.score(train_scaled, train_target)))
print("test score : " + str(ridge.score(test_scaled, test_target)))

# alpha를 이용해 규제를 약하게 혹은 강하게 할 수 있다.
# alpha 값 처럼 사람이 직접 넣어주는 값을 하이퍼파라미터 라고 한다.
# 적절한 alpha 찾기 : R^2 그래프 그려보기

import matplotlib.pyplot as plt
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge2 = Ridge(alpha = alpha)
    ridge2.fit(train_scaled, train_target)
    train_score.append(ridge2.score(train_scaled, train_target))
    test_score.append(ridge2.score(test_scaled, test_target))
    
plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# alpha = log10(x) = -1, x = 0.1이 최적임.
ridge_final = Ridge(alpha = 0.1)
ridge_final.fit(train_scaled, train_target)
print("\nPoly2 -> Ridge -> Optimal Alpha Linear score : ")
print("train score : " + str(ridge_final.score(train_scaled, train_target)))
print("test score : " + str(ridge_final.score(test_scaled, test_target)))

# Lasso 도 같은 방식으로 ㄱㄱ
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print("\nPoly2 -> Lasso -> Linear score : ")
print("train score : " + str(lasso.score(train_scaled, train_target)))
print("test score : " + str(lasso.score(test_scaled, test_target)))

train_score2 = []
test_score2 = []
alpha_list2 = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha2 in alpha_list2:
    lasso2 = Lasso(alpha = alpha2)
    lasso2.fit(train_scaled, train_target)
    train_score2.append(lasso2.score(train_scaled, train_target))
    test_score2.append(lasso2.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list2),train_score2)
plt.plot(np.log10(alpha_list2),test_score2)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# alpha = log10(x) = 1, x=10이 최적임
lasso_final = Lasso(alpha = 10)
lasso_final.fit(train_scaled, train_target)
print("\nPoly2 -> Lasso -> Optimal Alpha Linear score : ")
print("train score : " + str(lasso_final.score(train_scaled, train_target)))
print("test score : " + str(lasso_final.score(test_scaled, test_target)))

# 번외) lasso 와 ridge 동시에 출력하기
plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.plot(np.log10(alpha_list2),train_score2)
plt.plot(np.log10(alpha_list2),test_score2)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()