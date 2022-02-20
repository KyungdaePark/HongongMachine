import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1,100*100)

from sklearn.decomposition import PCA
pca = PCA(n_components = 50)
pca.fit(fruits_2d)

# PCA의 주성분 값을 표현해보자

import matplotlib.pyplot as plt


def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n / 10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols * ratio, rows * ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

draw_fruits(pca.components_.reshape(-1,100,100))

# 이제 원본 데이터를 TRANSFORM 해서 데이터의 차원을 50으로 바꿔보자.
fruits_pca = pca.transform(fruits_2d)

# 원상복귀 ?
fruits_inverse = pca.inverse_transform(fruits_pca)

# 원상복귀한 데이터를 그려보자.
for start in [0,100,200]:
    draw_fruits(fruits_inverse.reshape(-1,100,100)[start:start+100])
    print("\n")

# 조금 흐리게 나오지만 그래도 큰 차이가 없다.

# 주성분이 원본 데이터의 분산을 얼마나 잘 따라가는지를 나타낸 값을 "설명된 분산" 이라고 함.
# 설명된 분산은 explained_variance_ratio_임
print("PCA 의 설명된 분산(합) :")
print(np.sum(pca.explained_variance_ratio_))
plt.plot(pca.explained_variance_ratio_)
plt.show()
# 92% 잘 설명하구나.


# 이제 원본 데이터와 축소한 데이터를 지도 학습에 적용해 보고 어떤 차이가 있는지 알아보자
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
target = np.array([0]*100 + [1]*100 + [2]*100)

# a. 원본 데이터
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print("\n원본 데이터의 테스트 점수와 훈련 시간 : ")
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# b. 축소 데이터
scores = cross_validate(lr, fruits_pca, target)
print("\n축소 데이터의 테스트 점수와 훈련 시간 : ")
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# 시간 단축과 정확도가 향상된다.
# n_components 에 원하는 "설명된 분산"의 비율을 넣을  수 있음.
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
print("\n 설명된 분산이 50%만 있을 경우 차원의 개수 : ")
print(pca.n_components_)

# 단 2개의 주성분으로 원본 데이터를 변환해보자
fruits_pca2 = pca.transform(fruits_2d)

# 점수는?
scores = cross_validate(lr, fruits_pca2, target)
print("\n 설명된 분산이 50% (주성분이 2개)일 때의 테스트 점수와 훈련 시간 :")
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# 이제 이 주성분이 2개인 데이터를 이용해 k-평균 알고리즘으로부터 최적의 클러스터를 찾아보자.
# 그리고 그 클러스터로 원본 데이터를 비슷한 것들끼리 묶어 결과를 도출하자.

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca2)
for label in range(0,3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")

# 산점도를 그려보자. 차원이 2개밖에 없으니까 가능
for label in range(0,3):
    data = fruits_pca2[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['pineapple', 'banana', 'apple'])
plt.show()
