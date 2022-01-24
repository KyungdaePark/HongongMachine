# 훈련 세트의 점수가 테스트 세트의 점수보다 낮은 것이 문제 (과소적합)
# 가로,세로,두께를 이용해서 훈련하기 : 특성공학(feature engineering)

# perch의 가로 세로 두께 데이터를 가져옴
from math import degrees
import pandas as pd
df = pd.read_csv("http://bit.ly/perch_csv_data")
perch_full = df.to_numpy()
print(perch_full)

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

