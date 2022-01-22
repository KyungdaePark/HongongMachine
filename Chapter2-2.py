#  length 25, weight 150인 물고기가 1이 아닌 0으로 판독되는 오류가 생김
#  표준편차 (전처리, preprocessing)를 이용해 정확한 결과값 도출을 기대함


fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

from tokenize import PlainToken
import numpy as np

# column_stack 은 매개변수 a와 b의 n번째 요소를 각각 가져와 하나로 만듬
# concatenate는 1*35와 0*14의 배열을 한 배열로 합침.
fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# fish_data 와 fish_target을 이용해 train input,target 과 test input,target을 만듬
from sklearn.model_selection import train_test_split # train & test를 split

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, random_state = 49, stratify = fish_target
) # random_state 는 random seed, stratify는 target 비율에 맞추어(0과 1의 비율)
  # input 과 target의 비율을 정함

# 이제 input이 [25,150]인 경우가 정상적으로 판단할 수 있도록 할 것임. 


# 우선 [25,150]과 이웃이라고 판단한 원소들을 그래프로 표현해보자.
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
distances, indexes = kn.kneighbors([[25,150]])
# distances에는 [25,150]과 가장 가까운 k(=5)개와의 거리가 각각 들어감
# indexes에는 [25,150]과 가장 가까운 k(=5)개와의 원소들의 index가 각각 들어감

import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25,150,marker = '^')
plt.xlim(0,1000) # x좌표의 scale을 0부터 1000으로 둠
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker = 'D')
plt.show()


# 이를 해결하기 위해서는 data preprocessing(전처리)를 해 문제를 해결함.
# mean은 평균, std는 표준편차임. axis가 0이면 열을 따라서 ( 세로로 ) 계산함.
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)

train_scaled = (train_input - mean) / std # 표준점수를 구한다. 넘파이의 브로드캐스팅 기능으로 자동으로 배열 원소의 index에 맞게 계산해줌.
kn.fit(train_scaled, train_target)

new = ([25,150] - mean) / std # [25,150]에 대한 표준점수
re_distances, re_indexes = kn.kneighbors([new]) # 거리와 index를 다시구함
plt.scatter(train_scaled[:,0], train_scaled[:,1]) 
plt.scatter(new[0], new[1] ,marker = '^')
plt.scatter(train_scaled[re_indexes,0], train_scaled[re_indexes,1], marker = 'D') # 새로 이웃을 구함
plt.show()

test_scaled_input = (test_input - mean) / std
score = kn.score(test_scaled_input, test_target)
print(score)
